# ThinkBayes2

*Think Bayes* is an introduction to Bayesian statistics using computational methods.  

This is the repository for the forthcoming second edition; it is a work in progress.  If you are reading the first edition of the book, you don't want the code in this repo, yet.  Instead, you should go to [the repo for the first edition](https://github.com/AllenDowney/ThinkBayes).

You can run the code in this book on Binder by pressing this button:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/AllenDowney/ThinkBayes2/master)


The premise of this book, and the other books in the *Think X* series, is that if you know how to program, you can use that skill to learn other topics.

Most books on Bayesian statistics use mathematical notation and present ideas in terms of mathematical concepts like calculus. This book uses Python code instead of math, and discrete approximations instead of continuous mathematics. As a result, what would be an integral in a math book becomes a summation, and most operations on probability distributions are simple loops.

I think this presentation is easier to understand, at least for people with programming skills. It is also more general, because when we make modeling decisions, we can choose the most appropriate model without worrying too much about whether the model lends itself to conventional analysis. Also, it provides a smooth development path from simple examples to real-world problems.

*Think Bayes* is a Free Book. It is available under the [Creative Commons Attribution-NonCommercial 3.0 Unported License](https://creativecommons.org/licenses/by-nc/3.0/), which means that you are free to copy, distribute, and modify it, as long as you attribute the work and don’t use it for commercial purposes.

Other Free Books by Allen Downey are available from [Green Tea Press](https://greenteapress.com/wp/).

Note: The code is a version ahead of the book. I have not started revising the book yet.

## Getting started

To run the examples and work on the exercises in this book, you have to:

1.  Copy my files onto your computer.

2.  Install Python on your computer, along with the libraries we will
    use.

3.  Run Jupyter, which is a tool for running and writing programs, and
    load a **notebook**, which is a file that contains code and text.

The next three sections provide details for these steps. 

### Copying my files

The code for this book is available from
this **Git repository**. Git is a software tool that helps you keep track of the
programs and other files that make up a project. A collection of files
under Git's control is called a repository (the cool kids call it a
"repo"). GitHub is a hosting service that provides storage for Git
repositories and a convenient web interface.

Before you download these files, I suggest you copy my repository on
GitHub, which is called **forking**. If you don't already have a GitHub
account, you'll need to create one.

On this home page you should see a gray button
in the upper right that says Fork. If you press it, GitHub will create a
copy of my repository that belongs to you.

Now, the best way to download the files is to use a **Git client**,
which is a program that manages Git repositories. [Here are installation instructions for Windows, macOS, and Linux](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

In Windows, I suggest you accept the options recommended by the
installer, with two exceptions:

*   As the default editor, choose instead of .

*   For "Configuring line ending conversions", select "Check out as is, commit as is".

For macOS and Linux, I suggest you accept the recommended options.

Once the installation is complete, open a command window. On Windows,
open Git Bash, which should be in your Start menu. On macOS or Linux,
you can use Terminal.

To find out what directory you are in, type `pwd`, which stands for "print
working directory". On Windows, most likely you are in `Users\yourusername`. On MacOS or
Linux, you are probably in your home directory, `/home/yourusername`.

The next step is to copy files from your repository on GitHub to your
computer; in Git vocabulary, this process is called **cloning**. Run
this command:

```
git clone https://github.com/YourGitHubUserName/ThinkBayes2
```

Of course, you should replace `YourGitHubUserName` with your GitHub user name. After cloning,
you should have a new directory called `ThinkBayes2`.

If you don't want to use Git, you can [download my files in a Zip archive](https://github.com/AllenDowney/ThinkBayes2/archive/master.zip). You will need a program like `WinZip` or
`gzip` to unpack the Zip file. Make a note of the location of the files
you download.

### Installing Anaconda

You might already have Python installed on your computer, but you might
not have the latest version. To use the code in this book, I recommend
Python 3.6 or later. Even if you have the latest version, you probably
don’t have all of the libraries we need.

You could update Python and install these libraries, but I strongly
recommend that you don’t go down that road. I think you will find it
easier to use **Anaconda**, which is a free Python distribution that
includes all the libraries you need for this book (and more).

Anaconda is available for Linux, macOS, and Windows. By default, it puts
all files in your home directory, so you don’t need administrator (root)
permission to install it, and if you have a version of Python already,
Anaconda will not remove or modify it.

Start at [the Anaconda installation page](https://conda.io/docs/user-guide/install/index.html).
Download the installer for
your system and run it. You don’t need administrative privileges to
install Anaconda, so I recommend you run the installer as a normal user,
not as administrator or root.

I suggest you accept the recommended options. On Windows you have the
option to install Visual Studio Code, which is an interactive
environment for writing programs. You won’t need it for this book, but
you might want it for other projects.

The next step is to create a Conda environment that contains the packages
you need.  Open a command window and run the following commands:

```
cd ThinkBayes2
conda env create -f environment.yml
```

You might get a few error messages about packages that are not installed, but
we will not need them.

To activate the environment you just created, run

```
conda activate ThinkBayes2
```

To test whether the installation was successful, run

```
python install_test.py
```

If all is well, a window should appear with a graph.

When you are done working on this book, you might want to deactivate the environment:

```
conda deactivate
```

But when you want to work on this book again, you will have to activate the environment again.

If you prefer not to work with Conda environments, you could install the packages you need in the Conda "base" environment.  If you run the following commands in the `ThinkBayes2` directory, you should get everything you need:

```
conda install pandas jupyterlab seaborn
pip install .
```


### Running Jupyter

The code for each chapter, and starter code for the exercises, is in
Jupyter notebooks. If you have not used Jupyter before, you can [read about it here](https://jupyter.org).

To start Jupyter, open a command window (on macOS or Linux, open a Terminal; on Windows, open
Git Bash) and run the following commands:

```
cd ThinkBayes2
jupyter notebook
```

Jupyter should open a window in a browser, and you should see a list of directories.
Click on `notebooks` to open the directory containing the notebooks.  Then click on the first notebook; it should
open in a new tab.

In the notebook, press Shift-Enter to run the first few "cells". The first time you run a
notebook, it might take several seconds to start, while some Python
files get initialized. After that, it should run faster.

You can also launch Jupyter from the Start menu on Windows, from the Dock on
macOS, or from the Anaconda Navigator on any system. If you do that, Jupyter
might start in your home directory or somewhere else in your file
system, so you might have to navigate to find the `ThinkBayes2` directory.

I hope these instructions help you get started easily.  Please let me know if there is anything I can do to improve them.
