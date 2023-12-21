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

./run.sh --start-stage 2 --stop-stage 2

### Real and generated mel-spectrogram

- Real
![real melspectrogram](pics/2085003136_145600.png "real melspectrogram")

## Inference

./run.sh --start-stage 3 --stop-stage 3
