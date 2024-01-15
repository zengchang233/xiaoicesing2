# [XiaoiceSing2](https://www.isca-speech.org/archive/interspeech_2023/chunhui23_interspeech.html)
The source code for the paper [XiaoiceSing2](https://www.isca-speech.org/archive/interspeech_2023/chunhui23_interspeech.html) (interspeech2023)

[Demo page](https://wavelandspeech.github.io/xiaoice2/)

## Notice

I am busy with job-hunting now. I will update other modules, including the [HiFi-WaveGAN](https://arxiv.org/abs/2210.12740) after my final decision.

## Implementation (developping)

- [x] fastspeech2-based generator
- [x] discriminator group, including segment discriminators and detail discriminators
- [ ] ConvFFT block

## Dataset and preparation

- [x] opencpop ![cn](https://raw.githubusercontent.com/gosquared/flags/master/flags/flags/shiny/24/China.png)
- [ ] kiritan ![jp](https://raw.githubusercontent.com/gosquared/flags/master/flags/flags/shiny/24/Japan.png)
- [ ] CSD ![kr](https://raw.githubusercontent.com/gosquared/flags/master/flags/flags/shiny/24/South-Korea.png)
- [ ] m4singer ![cn](https://raw.githubusercontent.com/gosquared/flags/master/flags/flags/shiny/24/China.png)
- [ ] NUS48E 

Kaldi style preparation

- wav.scp
- utt2spk
- spk2utt
- text

```
./run.sh --start-stage 1 --stop-stage 1 # extract melspectrogram, f0, energy, and statistical value
```

## Training

```
./run.sh --start-stage 2 --stop-stage 2
```

### Real and generated melspectrogram (145600 training steps)

Real(left)  XiaoiceSing(middle)  XiaoiceSing2(right)

<div style="display:inline-block">
  <img src="pics/2085003136_145600.png" alt="real" width="250">
  <img src="pics/xs1_before_2085003136_145600.png" alt="xs1" width="250">
  <img src="pics/before_2085003136_145600.png" alt="xs2" width="250">
</div>

### L2 loss curve for melspectrogram

L2 loss before post-processing(left)    L2 loss after post-processing(right)

<div style="display:inline-block">
  <img src="pics/before_mel_l2_loss.png" alt="before" width="350">
  <img src="pics/post_mel_l2_loss.png" alt="after" width="350">
</div>

## Inference

```
./run.sh --start-stage 3 --stop-stage 3
```
