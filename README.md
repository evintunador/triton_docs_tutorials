# Triton Tutorials
making the [official triton documentation tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) actually comprehensible by *heavily* commenting in-detail about every little thing that's happening. Follow them in order of filename and check out the accompanying videos:

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/FtmnriHLbAg/0.jpg)](https://www.youtube.com/playlist?list=PL_NMDNzkCbLR69QafQUa6yTx-T-VPIoaZ)

*Note:* these tutorials were all tested and benchmarked on an Nvidia RTX 4060 Ti. On different GPUs your mileage may vary, and on GPUs with less VRAM or SRAM you may even receive errors. I've also found older GPUs running the exact same Triton code to get incorrect values (eg. RTX 3090) so I recommend using at least a 40 series

## learning resources I used
- of course the [official Triton documentation](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [here](https://github.com/hkproj/triton-flash-attention)'s a flash-attention implementation by one of my fav youtubers that comes with an [8 hour video](https://www.youtube.com/watch?v=zy8ChVd_oTM&t=1s)
- and the original flash-attention papers [v1](https://arxiv.org/abs/2205.14135) & [v2](https://arxiv.org/abs/2307.08691) (you only really need v2)
- [here](https://github.com/gpu-mode/lectures/tree/main
)'s a wider set of GPU kernel guides that includes an intro to Triton in lesson 14

