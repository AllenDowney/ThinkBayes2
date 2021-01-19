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

You don't need to know calculus or linear algebra.
You don't need any prior knowledge of statistics.  

In Chapter 1, I define
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


## Working with the code

Reading this book will only get you so far; to really understand it,
you have to work with the code.
The original form of this book is a series of Jupyter notebooks.
After you read each chapter, I encourage you to run the notebook and work
on the exercises.  
If you need help, my solutions are available.

There are several ways to run the notebooks:

* If you have Python and Jupyter installed, you can download the notebooks
    and run them on your computer.

* If you don't have a programming environment where you can run
    Jupyter notebooks, you can run them
    on Colab, which lets you run Jupyter notebooks in a browser without
    installing anything.

To run the notebooks on Colab, start from [this landing page](http://allendowney.github.io/ThinkBayes2/index.html),
which has links to all of the notebooks.

If you already have Python and Jupyter, you can
[download the notebooks as a Zip file](https://github.com/AllenDowney/ThinkBayes2/raw/master/ThinkBayes2Notebooks.zip).


## Installing Jupyter

If you don't have Python and Jupyter already, I recommend you
install Anaconda, which is a free Python distribution that includes all
the packages you'll need.
I found Anaconda easy to install.
By default it installs files in your home
directory, so you don't need administrator privileges. You can download
Anaconda from [this site](https://www.anaconda.com/products/individual).

By default Anaconda installs most of the packages you need to
run the code in this book.
But there are a few additional packages you need to install.

To make sure you have everything you need
(and the right versions), the best option is to create a Conda
environment.

```
        conda env create -f environment.yml
        conda activate ThinkBayes2
```

If you don't want to create an environment just for this book, you can
install what you need using Conda. The following commands should get
everything you need:

```
    conda install python jupyter pandas scipy matplotlib
    pip install empiricaldist
```

If you don't want to use Anaconda, you will need the following packages:

-   Jupyter to run the notebooks, <https://jupyter.org/>;

-   NumPy for basic numerical computation, <https://www.numpy.org/>;

-   SciPy for scientific computation, <https://www.scipy.org/>;

-   Pandas for working with data, <https://pandas.pydata.org/>;

-   matplotlib for visualization, <https://matplotlib.org/>;

-   empiricaldist for representing distributions, <https://pypi.org/project/empiricaldist/>. .

Although these are commonly used packages, they are not included with
all Python installations, and they can be hard to install in some
environments. If you have trouble installing them, I recommend using
Anaconda or one of the other Python distributions that include these
packages.




### Contributor List

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
    especially in Fall 2012, when I audited his class on Bayesian
    Inference at Olin College.

-   I wrote parts of this book during project nights with the Boston
    Python User Group, so I would like to thank them for their company
    and pizza.

-   Jasmine Kwityn and Dan Fauxsmith at O'Reilly Media proofread the
    first edition and found many opportunities for improvement.

-   Linda Pescatore found a typo and made some helpful suggestions.

-   Tomasz Miasko sent many excellent corrections and suggestions.

Other people who spotted typos and small errors include
Greg Marra,
Matt Aasted,
Marcus Ogren,
Tom Pollard,
Paul A. Giannaros, Jonathan Edwards, George Purkins, Robert Marcus, Ram
Limbu, James Lawry, Ben Kahle, Jeffrey Law,
Alvaro Sanchez,
Olivier Yiptong,
Yuriy Pasichnyk,
Kristopher Overholt,
Max Hailperin,
Markus Dobler
