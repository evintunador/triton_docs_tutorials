# triton_docs_tutorials **(WIP)**
making the [official triton documentation tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) actually comprehensible by *heavily* commenting in-detail about every little thing that's happening. Check out the accompanying videos:

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/TUQAyCNxFe4/0.jpg)](https://youtube.com/playlist?list=PLPefVKO3tDxOJLAmCA75uShbe1z_RNqkQ&si=C5VF9fNW8CYZzh9x)

## how to get up and running
these instructions are for setting up a cloud GPU instance on [vast.ai](https://vast.ai); other cloud providers should be similar. If you're running linux on your own pc with your own GPU then you can skip all this

1. setup an account and input your payment information on [vast.ai](https://vast.ai), [lambdalabs](https://lambdalabs.com) or similar provider. You can compare prices [here](https://cloud-gpus.com)
2. launch a GPU instance. You can choose whichever single GPU is cheapest (I can always find one for less than $1/hr), but I'd recommend at least 16GB of RAM to ensure that all of the tests and benchmarks in this repo run without overwhelming memory.
3. once it's running, open the included remote jupyter lab environment (Vast & Lambda provide this so i presume others do too). Optionally you could instead ssh into the instance in order to be able to use your own IDE
4. Open the terminal in your instance and update everything jic
```
sudo apt update
```
```
sudo apt install build-essential
```
5. install github CLI
```
sudo apt install gh
```
5. Input the following command and follow the prompts to log in to your GitHub account
```
gh auth login
```
6. Clone your fork of my repository
```
gh repo clone your_github_username/triton_docs_tutorials
```
7. setup your git user email and username
```
git config --global user.email "your_github_email@email.com"
```
```
git config --global user.name "your_github_username"
```
8. now you can make changes and push updates as usual all through the jupyterlab environment. Note that unless you also setup a filesystem before intializing your GPU instance that everything will be deleted when you close out the instance, so don't forget to push your changes to github!
9. install all necessary packages
```
pip install numpy matplotlib pandas torch triton pytest
```
10. and force an update to them jic; using newer Triton versions can be very important if you're experiencing bugs
```
pip install --upgrade torch
```
```
pip install --upgrade triton
```

*note: if you've got an AMD GPU then the installation process should be the same, but throughout the repo you'll have to do your own research on the relatively small edits required to make your code more specifically efficient for your hardware. those edits can be found in the original official docs tutorials; i removed them from my version*

## learning resources I used
- of course the [official Triton documentation](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [here](https://github.com/hkproj/triton-flash-attention)'s a flash-attention implementation by one of my fav youtubers that comes with an [8 hour video](https://www.youtube.com/watch?v=zy8ChVd_oTM&t=1s)
- and the original flash-attention papers [v1](https://arxiv.org/abs/2205.14135) & [v2](https://arxiv.org/abs/2307.08691) (you only really need v2)
- [here](https://github.com/gpu-mode/lectures/tree/main
)'s a wider set of GPU kernel guides that includes an intro to Triton in lesson 14

