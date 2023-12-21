# xiaoicesing2
The source code for the paper XiaoiceSing2 (interspeech2023)

## Implementation

- [x] fastspeech2-based generator
- [x] discriminator group, including segment discriminators and detail discriminators
- [ ] ConvFFT block

## Dataset and preparation

- [x] opencpop
- [ ] kiritan
- [ ] m4singer
- [ ] NUS48E

## Training

```
./run.sh --start-stage 2 --stop-stage 2
```

### Real and generated melspectrogram

- Real

![real melspectrogram](pics/2085003136_145600.png "real melspectrogram")

- before post-processing

![before melspectrogram](pics/before_2085003136_145600.png)

- after post-processing

![after melspectrogram](pics/after_2085003136_145600.png)

### L2 loss curve for melspectrogram

- L2 loss before post-processing

![L2 loss before](pics/before_mel_l2_loss.png)

- L2 loss after post-processing

![L2 loss after](pics/post_mel_l2_loss.png)

## Inference

```
./run.sh --start-stage 3 --stop-stage 3
```
