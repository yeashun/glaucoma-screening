"""Single-model serial prediction; cost.py calls this same Predictor."""
import argparse
import time
from common import *
from glaucoma.data import open_rgb, valid_bbox, image_tensor, encode_age


class Predictor:
    def __init__(self, checkpoint, device='cuda'):
        self.device = torch.device(device)
        self.model, self.state = load_model(checkpoint, self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, path, row):
        image = open_rgb(path)
        image = image_tensor(image.crop(valid_bbox(image)), self.state['config']['image_size'])
        age = encode_age(row, self.state['age_stats'])
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
            logit = self.model(image[None].to(self.device), age[None].to(self.device))
        probability = float(logit.float().sigmoid().item())
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        return probability


def predict(checkpoint, data_root, output):
    torch.set_num_threads(4)
    root, output = Path(data_root), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Never overwrite an existing submission')
    rows = read_rows(root, 'test')
    if len(rows) != 5855:
        raise ValueError('Unexpected test count')
    paths = {p.name.casefold(): p for p in (root / 'test').iterdir() if p.is_file()}
    predictor = Predictor(checkpoint)
    temp = output.with_suffix('.csv.partial')
    started = time.monotonic()
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
        for i, row in enumerate(rows):
            score = predictor(paths[row['Image_name'].casefold()], row)
            writer.writerow([row['Image_name'], format(score, '.9g'), int(score >= .5)])
            if (i + 1) % 100 == 0:
                f.flush()
                print(json.dumps({'event': 'test_prediction', 'images': i+1, 'total': len(rows)}), flush=True)
    with temp.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['Image_name', 'Probability', 'Pred_Label']:
            raise ValueError('Invalid submission columns')
        predicted = list(reader)
    if [r['Image_name'] for r in predicted] != [r['Image_name'] for r in rows]:
        raise ValueError('Missing/duplicate/misordered predictions')
    if any(not 0 <= float(r['Probability']) <= 1 or int(r['Pred_Label']) != int(float(r['Probability']) >= .5)
           for r in predicted):
        raise ValueError('Invalid probabilities or labels')
    temp.replace(output)
    write_json(output.with_suffix('.metadata.json'),
               {'rows': len(rows), 'checkpoint_sha256': sha256(checkpoint), 'submission_sha256': sha256(output),
                'test_csv_sha256': sha256(root / 'test.csv'), 'precision': 'fp16', 'batch_size': 1,
                'gpu': torch.cuda.get_device_name(), 'elapsed_s': time.monotonic()-started,
                'positive_predictions': sum(int(r['Pred_Label']) for r in predicted)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    predict(args.checkpoint, args.data_root, args.output)
