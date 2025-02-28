## how to get up and running
these instructions are for setting up a cloud GPU instance on [vast.ai](https://vast.ai); other cloud providers should be similar. If you're running linux on your own pc with your own GPU then you can skip all this

see also the accompanying video:
[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/mmRlZKFLAvE/0.jpg)](https://www.youtube.com/watch?v=mmRlZKFLAvE)

1. setup an account and input your payment information on [vast.ai](https://vast.ai), [lambdalabs](https://lambdalabs.com) or similar provider. You can compare prices [here](https://cloud-gpus.com)
2. launch a GPU instance. You can choose whichever single GPU is cheapest (I can always find one for less than $1/hr on lambda and less than $0.15/hr on vast), but I'd recommend at least 16GB of RAM to ensure that all of the tests and benchmarks in this repo run without overwhelming memory. 
    - *Note:* I'd also recommend choosing a GPU with at least the [Ampere architecture](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) or even newer. I've found the exact same Triton code can sometimes provide incorrect final values on older GPUs; not sure if this is a bug with Triton or a limitation of older GPU hardware. All the tests in this repo were run on an 4060Ti
3. once it's running, open the included remote jupyter lab environment (Vast & Lambda provide this so i presume others do too). Optionally you could instead ssh into the instance in order to be able to use your own IDE, but I'll let chatGPT help you if you want to do that
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
6. Clone your fork of this repository
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
8. now you can make changes and push updates as usual all through the jupyterlab environment's terminal. Note that unless you also setup a filesystem before intializing your GPU instance that everything will be deleted when you close out the instance, so don't forget to push your changes to github!
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
11. Once you're done with all changes pushed, make sure to logout so some random GPU provider doesn't have access to your github account
```
gh auth logout
```

*note: if you're on an AMD GPU then this whole process should likely be the same, but throughout the repo you'll have to do your own research on the relatively small edits required to make your code more specifically efficient for your hardware. those edits can be found in the [original official triton docs tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html); i removed them from my version*