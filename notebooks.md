# Introduction to notebooks

A **Jupyter notebook** is a file that holds text, code, math, and the results of computations, including graphics. You can edit the notebook in a browser. Code is executed on demand by a *kernel* running a session of Python or other interactive language environment.

A notebook is divided into *cells*. Each cell may hold text/math or code, but not both. You can switch the type of a cell by menus or shortcuts. You can also move them up and down, etc.

## Code
When a code cell is executed (play button or C-Enter), it may have associated output that appears beneath it. The executed cells are numbered consecutively. 

```{warning}
Unlike a program or script, the execution order of notebook cells, which is shown by their numbering, is independent of the top-to-bottom ordering of the cells.  
```

```{tip}
If you want to execute all cells from top to bottom, you can select "Run all cells" from a menu. When you do that, the current variable and function definitions are not cleared before execution. To do that, you have to restart the kernel first. Restarting and then running all cells is the only way to experience the notebook like someone who is opening it for the first time.
```

## Text

Text is expressed using *Markdown* for words and *LaTeX* for math. When such a cell is "executed," it is rendered. It does not affect the kernel in any way.

Markdown is easy. Help on Markdown syntax is found under Help/Markdown. A blank line starts a new paragraph.

Entering math via LaTeX is more complicated. To put math within a line of text, enclose it in dollar signs. To make a displayed equation, use double dollar signs. You can use `_` for subscripts and `^` for superscripts. If more than one character is to be sub or super, put them inside curly braces. You make a fraction using `\frac{numer}{denom}`. There are sums and integrals using `\sum` and `\int`. Enter common functions like `\sin` and `\cos`. A Greek letter also starts with a backslash, e.g., `\theta`. 

That should be enough for most things. 

## Saving work

Every few minutes, a notebook automatically creates a *checkpoint*. This is a temporary autosave state that allows you to recover from mistakes during the session. If you want to save the notebook state permanently, choose File/Save.

Under the File menu, you can choose "Save and export" to get different static snapshots of the notebook. HTML is fine for screens (although many people won't know how to view it). If you want to save as PDF, the "PDF" option might require installing additional software. Try "Webpdf" instead. If that also doesn't work for some reason, select File/Print and then save as PDF from the print dialog.


