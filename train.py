# encoding=utf-8
from model import *


class InpaintDataset(td.Dataset):
    def __init__(self, tag: str):
        self.tag = tag
        self.ndsi_norm = 10000
        self.temp_bias = 273.15
        self.temp_norm = 35
        self.dem_bias = 8678 / 2.
        self.dem_norm = self.dem_bias
        self.dtype = torch.float32

        self.data_dir = packed_train_data_dir
        self.data_len = len(list(self.data_dir.glob('*.pt')))
        self.mask_dir = sampled_mask_dir
        self.mask_len = len(list(self.mask_dir.glob('*.pt')))

        self.env_z = torch.load(environment_z_dir / (tag + '_z.pt')).to(self.dtype)

    def __getitem__(self, file_idx: int):
        mask = torch.load(self.mask_dir / f'{np.random.randint(0, self.mask_len):07d}.pt').cuda()

        batch_data = torch.load(self.data_dir / f'{file_idx:07d}.pt').to(self.dtype).cuda()
        z = self.env_z[file_idx][:, 1:4].cuda()

        # transformation
        batch_data[:, 0:2] /= self.ndsi_norm
        batch_data[:, 2] = (batch_data[:, 2] - self.dem_bias) / self.dem_norm
        batch_data[:, 3] = (batch_data[:, 3] - self.temp_bias) / self.temp_norm

        label = torch.cat([batch_data[:, 0:1], batch_data[:, 2:4]], dim=1)

        interp_ndsi = batch_data[:, 1:2] * mask + batch_data[:, 0:1] * ~mask
        train_data = torch.cat([mask, interp_ndsi, batch_data[:, 2:4]], dim=1)

        return train_data, label, mask.float(), z       # batch_data[:, 1:2]

    def __len__(self):
        return self.data_len


def train(model_str,
          max_epoch: int = 30, learning_rate=5e-4, adamw_beta=(0.9, 0.95),
          log_interval: int = 200, ckpt_interval: int = 500, grad_norm: float = 2.0, **kwargs):
    data_loader = td.DataLoader(InpaintDataset(tag='Train'),
                                batch_size=1, shuffle=True, num_workers=0)
    model = model_dict[model_str](**kwargs).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=adamw_beta)

    for epoch in range(max_epoch):
        print(f'Epoch {epoch} start')
        loss_lst, time_lst = [], []

        batch_start_time = time.time()
        for idx, (train_data, label, label_mask, z) in enumerate(data_loader):
            # train_data: shape is (batch_size, 4, pic_size, pic_size)
            # label: shape is (batch_size, 1, pic_size, pic_size)
            # label_mask: shape is (batch_size, 1, pic_size, pic_size)
            # z: shape is (batch_size, 3)
            train_data, label, label_mask, z = train_data[0], label[0], label_mask[0], z[0]

            # train Generator
            optimizer.zero_grad()
            output = model(train_data, z)
            loss = loss_func(label, output, label_mask)
            loss.backward()
            if grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
            optimizer.step()
            loss_lst.append(loss.item())

            # print log
            time_lst.append(time.time() - batch_start_time)
            batch_start_time = time.time()
            if idx % log_interval == 0 and idx > 0:
                last_loss = np.array(loss_lst[-log_interval:]).mean()
                last_time = np.array(time_lst[-log_interval:]).mean()
                output_str = f'\tIDX {idx:06d}, loss: {last_loss :.4f}, time cost: {last_time:.3f}s'
                print(output_str)

            # save checkpoint
            if idx % ckpt_interval == 0 and idx > 0:
                ckpt_file = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict()
                }
                torch.save(ckpt_file, ckpt_save_dir / f'ckpt_epoch_{epoch:02d}_batch_{idx:06d}.pt')
                break
        break


class Evaluator:
    def __init__(self):
        self.sum_diff = 0
        self.sum_ssr = 0
        self.sum_sst = 0
        self.num = 0

    def add(self, gt: torch.Tensor, pred: torch.Tensor, eval_mask: torch.Tensor):
        assert gt.shape == pred.shape, 'gt shape {} with pred shape {}'.format(gt.shape, pred.shape)

        diff = (gt - pred)[eval_mask]
        eval_num = torch.sum(eval_mask).item()

        if eval_num > 0:
            gt_mean = torch.sum(gt * eval_mask) / eval_num
            e2 = torch.sum(torch.square(diff)).item()
            self.sum_ssr += e2
            self.sum_sst += torch.sum(torch.square((gt - gt_mean) * eval_mask)).item()
            self.sum_diff += e2
            self.num += eval_num

    def get(self):
        return {'rmse': (self.sum_diff / self.num) ** 0.5 if self.num > 0 else 0, 'N': self.num,
                'r2': 1 - self.sum_ssr / self.sum_sst if self.sum_sst > 0 else 0}


def test(model_str, dataset_tag, ckpt_epoch, ckpt_batch, **kwargs):
    model = model_dict[model_str](**kwargs).cuda()
    ckpt_path = ckpt_save_dir / f'ckpt_epoch_{ckpt_epoch:02d}_batch_{ckpt_batch:06d}.pt'
    if ckpt_path.exists():
        print('Load:', ckpt_path.stem)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    evaluator = Evaluator()
    assert dataset_tag in ['Test', 'Strict_Test']
    data_loader = td.DataLoader(InpaintDataset(tag=dataset_tag), batch_size=1, shuffle=False, num_workers=0)

    for idx, (input_data, label, label_mask, z) in enumerate(data_loader):
        input_data, label, label_mask, z = input_data[0], label[0], label_mask[0], z[0]
        with torch.no_grad():
            output = model(input_data, z)[:, :1]
        label = label[:, :1]
        label[label.lt(0)] = 0
        output[output.lt(0)] = 0
        evaluator.add(gt=label * 100, pred=output * 100, eval_mask=label_mask.gt(0))
        if idx % 512 == 0:
            print(evaluator.get())
        break
    print('total:', evaluator.get())


def main():
    print('Start running.')
    # training
    # train('MAT_MS')

    # testing
    # ckpt_epoch, ckpt_batch = 27, 10000
    # test('MAT_MS', 'Test', ckpt_epoch=ckpt_epoch, ckpt_batch=ckpt_batch)
    # test('MAT_MS', 'Strict_Test', ckpt_epoch=ckpt_epoch, ckpt_batch=ckpt_batch)


if __name__ == '__main__':
    root_data_dir = Path('E:/HRY/Data')
    packed_train_data_dir = root_data_dir / 'Train'
    sampled_mask_dir = root_data_dir / 'Mask'
    environment_z_dir = root_data_dir / 'idx_for_building'
    ckpt_save_dir = Path('./checkpoint')
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    main()
