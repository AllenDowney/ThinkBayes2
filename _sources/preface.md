# Preface

The premise of this book, and the other books in the *Think X* series,
is that if you know how to program, you can use that skill to learn
other topics.

Most books on Bayesian statistics use math notation and present
ideas using mathematical concepts like calculus. This book uses
Python code and discrete approximations instead of
continuous mathematics. As a result, what would be an integral in a math
book becomes a summation, and most operations on probability
distributions are loops or array operations.

I think this presentation is easier to understand, at least for people
with programming skills. It is also more general, because when we make
modeling decisions, we can choose the most appropriate model without
worrying too much about whether the model lends itself to mathematical
analysis.

Also, it provides a smooth development path from simple examples to
real-world problems.

## Modeling

Most chapters in this book are motivated by a real-world problem, so
they involve some degree of modeling.  Before we can apply Bayesian
methods (or any other analysis), we have to make decisions about which
parts of the real-world system to include in the model and which
details we can abstract away.

For example, in Chapter xxx, the motivating problem is to
predict the winner of a soccer (football) game.  I model goal-scoring as a
Poisson process, which implies that a goal is equally likely at any
point in the game.  That is not exactly true, but it is probably a
good enough model for most purposes.

In Chapter xxx the motivating problem is interpreting SAT
scores (the SAT is a standardized test used for college admissions in
the United States).  I start with a simple model that assumes that all
SAT questions are equally difficult, but in fact the designers of the
SAT deliberately include some questions that are relatively easy and
some that are relatively hard.  I present a second model that accounts
for this aspect of the design, and show that it doesn't have a big
effect on the results after all.

I think it is important to include modeling as an explicit part
of problem solving because it reminds us to think about modeling
errors (that is, errors due to simplifications and assumptions
of the model).

Many of the methods in this book are based on discrete distributions,
which makes some people worry about numerical errors.  But for
real-world problems, numerical errors are almost always
smaller than modeling errors.

Furthermore, the discrete approach often allows better modeling
decisions, and I would rather have an approximate solution
to a good model than an exact solution to a bad model.

## Who is this book for?

To start this book, you should be comfortable with Python.
If you are familiar with NumPy and Pandas, that will help, but I'll
explain what you need as we go.

You don't need any prior knowledge of statistics.  In Chapter 1, I define
probability and introduce the idea of conditional probability, which is the
foundation of Bayes's Theorem.

Chapter 3 introduces the idea of a probability distribution, which is the
foundation of Bayesian statistics.

Along the way, we will use a variety of discrete and continuous distributions,
including the binomial, exponential, Poisson, beta, gamma, and normal
distributions.
I will explain each distribution when it is introduced, and we will use
SciPy to compute them, so you don't need to know anything about their
mathematical properties.

As a result, you don't need to know calculus or linear algebra.

## Working with the code

Reading this book will only get you so far; to really understand the material,
you have to work with the code.
The original form of this book is a series of Jupyter notebooks.
After you read each chapter, I encourage you to run the notebook and work
on the exercises at the end of each chapter.  If you can do the exercises,
you understand the material.  But if you need help, my solutions are
available.

There are several ways you can work with the code in this book:

-   If you don't have a programming environment where you can run
    Jupyter notebooks, and you don't want to create one, you can run the
    notebooks on Colab, which is an online service provided by Google.
    Colab let's you run Jupyter notebooks in a browser without
    installing anything.

-   If you have Python and Jupyter installed, you can download the notebooks
    and run them on your computer.

To run the notebooks on Colab, start from [](), 
which has links to all of the notebooks.

If you already have Python and Jupyter, you can download the code from
my Git repository, at <https://github.com/AllenDowney/ThinkBayes>. Git
is a version control system that allows you to keep track of the files
that make up a project. A collection of files under Git's control is
called a "repository". GitHub is a hosting service that provides storage
for Git repositories and a convenient web interface.

The GitHub homepage for my repository provides several ways to download
the code:

-   You can create a copy of my repository on GitHub by pressing the
    Fork button. If you don't already have a GitHub account, you'll need
    to create one. After forking, you'll have your own repository on
    GitHub that you can use to keep track of code you write while
    working on this book. Then you can clone the repo, which means that
    you copy the files to your computer.

-   Or you could clone my repository. You don't need a GitHub account to
    do this, but you won't be able to write your changes back to GitHub.

-   If you don't want to use Git at all, you can download the files in a
    Zip file using the button in the lower-right corner of the GitHub
    page. Or you can download the Zip file from []().

If you don't have Python and Jupyter installed already, I recommend you
install Anaconda, which is a free Python distribution that includes all
the packages you'll need to run the code (and lots more). I found
Anaconda easy to install. By default it installs files in your home
directory, so you don't need administrator privileges. You can download
Anaconda from <https://www.anaconda.com/products/individual>.

If you install Anaconda, you will have most of the packages you need to
run the code in this book. To make sure you have everything you need
(and the right versions), the best option is to create a Conda
environment. And the best way to do that is to use the command line. If
you are not familiar with the command line, you might want to run the
notebooks on Colab.

1.  After downloading my repository, you should have a directory named .
    Use to move into that directory.

2.  Use to confirm that you have a file named . It lists the packages
    you need.

3.  Run the following command to create an environment:

        conda env create -f environment.yml

4.  Run the following command to activate the environment you just
    created:

        conda activate ThinkBayes2

5.  To test your environment and make sure it has everything we need,
    run the following command:

        python test_env.py

If you don't want to create an environment just for this book, you can
install what you need using Conda. The following commands should get
everything you need:

    conda install python jupyter pandas scipy matplotlib
    pip install empiricaldist

If you don't want to use Anaconda, you will need the following packages:

-   Jupyter to run the notebooks, <https://jupyter.org/>;

-   NumPy for basic numerical computation, <http://www.numpy.org/>;

-   SciPy for scientific computation, <http://www.scipy.org/>;

-   Pandas for working with data, <https://pandas.pydata.org/>;

-   matplotlib for visualization, <http://matplotlib.org/>;

-   empiricaldist for representing distributions, [](); .

Although these are commonly used packages, they are not included with
all Python installations, and they can be hard to install in some
environments. If you have trouble installing them, I recommend using
Anaconda or one of the other Python distributions that include these
packages.


Prerequisites
-------------

There are several excellent modules for doing Bayesian statistics in
Python, including and OpenBUGS. I chose not to use them for this book
because you need a fair amount of background knowledge to get started
with these modules, and I want to keep the prerequisites minimal. If you
know Python and a little bit about probability, you are ready to start
this book.

Chapter [\[intro\]](#intro){reference-type="ref" reference="intro"} is
about probability and Bayes's theorem; it has no code.
Chapter [\[compstat\]](#compstat){reference-type="ref"
reference="compstat"} introduces , a thinly disguised Python dictionary
I use to represent a probability mass function (PMF). Then
Chapter [\[estimation\]](#estimation){reference-type="ref"
reference="estimation"} introduces , a kind of Pmf that provides a
framework for doing Bayesian updates.

In some of the later chapters, I use analytic distributions including
the Gaussian (normal) distribution, the exponential and Poisson
distributions, and the beta distribution. In
Chapter [\[species\]](#species){reference-type="ref"
reference="species"} I break out the less-common Dirichlet distribution,
but I explain it as I go along. If you are not familiar with these
distributions, you can read about them on Wikipedia. You could also read
the companion to this book, *Think Stats*, or an introductory statistics
book (although I'm afraid most of them take a mathematical approach that
is not particularly helpful for practical purposes).

Contributor List {#contributor-list .unnumbered}
----------------

If you have a suggestion or correction, please send email to
*downey\@allendowney.com*. If I make a change based on your feedback, I
will add you to the contributor list (unless you ask to be omitted).

If you include at least part of the sentence the error appears in, that
makes it easy for me to search. Page and section numbers are fine, too,
but not as easy to work with. Thanks!

-   First, I have to acknowledge David MacKay's excellent book,
    *Information Theory, Inference, and Learning Algorithms*, which is
    where I first came to understand Bayesian methods. With his
    permission, I use several problems from his book as examples.

-   This book also benefited from my interactions with Sanjoy Mahajan,
    especially in fall 2012, when I audited his class on Bayesian
    Inference at Olin College.

-   I wrote parts of this book during project nights with the Boston
    Python User Group, so I would like to thank them for their company
    and pizza.

-   Olivier Yiptong sent several helpful suggestions.

-   Yuriy Pasichnyk found several errors.

-   Kristopher Overholt sent a long list of corrections and suggestions.

-   Max Hailperin suggested a clarification in
    Chapter [\[intro\]](#intro){reference-type="ref" reference="intro"}.

-   Markus Dobler pointed out that drawing cookies from a bowl with
    replacement is an unrealistic scenario.

-   In spring 2013, students in my class, Computational Bayesian
    Statistics, made many helpful corrections and suggestions: Kai
    Austin, Claire Barnes, Kari Bender, Rachel Boy, Kat Mendoza, Arjun
    Iyer, Ben Kroop, Nathan Lintz, Kyle McConnaughay, Alec Radford,
    Brendan Ritter, and Evan Simpson.

-   Greg Marra and Matt Aasted helped me clarify the discussion of *The
    Price is Right* problem.

-   Marcus Ogren pointed out that the original statement of the
    locomotive problem was ambiguous.

-   Jasmine Kwityn and Dan Fauxsmith at O'Reilly Media proofread the
    book and found many opportunities for improvement.

-   Linda Pescatore found a typo and made some helpful suggestions.

-   Tomasz Miasko sent many excellent corrections and suggestions.

Other people who spotted typos and small errors include Tom Pollard,
Paul A. Giannaros, Jonathan Edwards, George Purkins, Robert Marcus, Ram
Limbu, James Lawry, Ben Kahle, Jeffrey Law, and Alvaro Sanchez.
