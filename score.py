import torch
import torch.nn.functional as F
from eval import eval


if __name__ == "__main__":
    weights_bass = torch.Tensor([2.33, 2.53, 3.6, 3.4, 3.17, 4.23, 3.37])
    weight_mid = torch.Tensor([2.53, 2.63, 3.63, 3.27, 2.5, 3.67, 2.97])
    weight_treble = torch.Tensor([2.37, 2.97, 3.67, 3.2, 2.93, 4, 3.07])
    weight_avg = (weights_bass + weight_mid+weight_treble) / 3

    output = eval(tag='./test/KAWAI.wav',  split_mode=True)

    if torch.cuda.is_available():
        output = output.cuda()
        weight_avg = weight_avg.cuda()

    if output.size(0) == weight_avg.size(0):
        score = float((F.softmax(output)[0] * weight_avg).sum())
        print('Score : ' + str(round(score, 2)))

    else:
        print('Class dimension mismatched, unable to score.')
