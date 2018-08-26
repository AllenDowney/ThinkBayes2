# ThinkBayes2

*Think Bayes* is an introduction to Bayesian statistics using computational methods.

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

## Copying my files

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

## Installing Python

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

By default, Anaconda installs most of the packages you need, but there
are a few more you might have to add. 

There are two ways to do that:

1.  Open a command window. On macOS or Linux, you can use Terminal. On
Windows, open the Anaconda Prompt that should be in your Start menu.
Run the following command (copy and paste it if you can, to avoid
typos):

```
conda install jupyterlab pandas seaborn
```

2.  Create a Conda environment with the packages you need.  You can do that using
the `environment.yml` file in this repository.

