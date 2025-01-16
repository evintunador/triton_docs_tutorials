# triton_docs_tutorials **(WIP)**
making the official triton tutorials actually comprehensible by *heavily* commenting in-detail about every little thing that's happening
https://triton-lang.org/main/getting-started/tutorials/index.html

## why triton?
You might be asking: why are we using Triton instead of CUDA? I'm open to the idea of learning and then doing a lesson on CUDA (and MPS for that matter) in the future, but for now here are the pros and cons that it came down to:

|      | triton                                                                                            | cuda                                                             |
| ---- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| pros | - written in Python (quicker to learn)<br>- works on more than just Nvidia GPUs<br>- open-sourced | - broadly used<br>- linux or windows                             |
| cons | - less commonly used<br>- requires linux                                                          | - written in C<br>- only works on Nvidia GPUs<br>- closed-source |

Personally I'm on a Mac so i plan on doing all my work on a cloud provider like [lambdalabs](https://lambdalabs.com) anyways so the windows availability didn't matter much to me. That and I highly value the pythonic syntax and potential future widespread compatibility. 

*note: if you've got an AMD GPU then the installation process should be the same, but throughout the repo you'll have to do your own research on the relatively small edits required to make your code more specifically efficient for your hardware. those edits can be found in the original official docs tutorials; i removed them from my version*

## how to get up and running
these instructions are for setting up a cloud GPU instance on [lambdalabs](https://lambdalabs.com); other cloud providers should be similar. If you're running linux on your own pc with your own GPU then it should be obvious which of these steps to skip. 

1. setup an account on [lambdalabs](https://lambdalabs.com) or similar provider, including putting in your payment info
2. launch a GPU instance. You can choose whichever single GPU is cheapest (I can always find one for less than $1/hr)
3. once it's running, open the jupyter lab environment (Lambda provides this; i presume others do too). Optionally you could instead ssh into the instance in order to be able to use your own IDE
4. Open the terminal in your instance and install github CLI
```
sudo apt update
```
```
sudo apt install gh
```
5. Input the following command and follow the prompts to log in to your GitHub account. Choose https and you'll need to open up the provided link in your browser
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
pip install numpy matplotlib pandas torch triton
```

## learning resources I used
here's someone else's helpful attempt at explaining the official docs but every fifth line from them was "idk what this does"
- https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c
- https://isamu-website.medium.com/understanding-triton-tutorials-part-2-f6839ce50ae7
here's a flash-attention implementation by one of my fav youtubers
- https://github.com/hkproj/triton-flash-attention
meta's xformers repo implements basically everything one could need
- https://github.com/facebookresearch/xformers
here's a wider set of GPU kernel guides that includes an intro to Triton in lesson 14
- https://github.com/gpu-mode/lectures/tree/main


