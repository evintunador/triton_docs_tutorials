# live-deriving Apple's cut cross-entropy loss (WIP)
As I've said multiple times throughout these lessons, watching my videos and copying my code line by line does not count as learning, or as having completed this course. Before you can claim that you know how to write GPU kernels, you need to go actually design and write one (or many, preferably many) from first principles. 

For that reason, this is not a "lesson" in the way that the prior 9 were. Rather, it's a demonstration of an example project that you should do to test your knowledge. One alternative way to think about them while you watch is that you, the viewer, are a prospective employer who just asked me a very hard job interview question and I'm now attempting to answer it to demonstrate my skills.

In the first video I start with a goal derived from a vague memory of when I skimmed through Apple's cut cross-entropy loss [paper](https://arxiv.org/abs/2411.09009) months before having ever even written my first GPU kernel. Using that loose starting intuition I work out a plan for what will hopefully be a relatively efficient fused CE Loss kernel from first principles. Then for the second video I attempt to take that plan and put it into action. 

Will it even run? Was there something I wasn't accounting for? Will it be as fast (or hopefully faster than) PyTorch? Does it even resemble Apple's algorithm or does it take a different route entirely? Idk, this was all live off-the-cuff so we'll see.

*the answer is that I could not in the length of the two videos get it working. if i feel like putting more effort into making it actually run (and even more to making it actually fast) then i'll update this readme accordingly and maybe even make a third explanatory video. but for now it doesn't work*

*videos below are placeholders until the actual vidoes get posted*

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/ftknUZDQCPc/0.jpg)](https://www.youtube.com/watch?v=ftknUZDQCPc)
[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/ftknUZDQCPc/0.jpg)](https://www.youtube.com/watch?v=ftknUZDQCPc)

BTW, if you're looking for an idea for a project to do, maybe try to not only fix this kernel but fuse the forward & backward pass into one single kernel. What I mean by that is instead of outputting the final loss value, you can have your kernel just skip that step and go straight to outputting $\frac{\partial L}{\partial x}$ and $\frac{\partial L}{\partial E}$. Then, if someone were to use your new kernel, instead of doing the traditional .backward() on the final loss value, they'd actually do it on x and manually accumulate to E using the gradients you gave them. I've not actually done this myself, but I'm vaguely under the impression that this is what [Liger Kernel](https://github.com/linkedin/Liger-Kernel) does from one time when I skimmed a few lines of the wrapper function around their kernel.