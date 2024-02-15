# Function
This script will calculate probability density functions defined by a median, upper and lower quartiles provided by a user. It will attempt to calculate a skew normal, gamma and Weibull distribution.

# Getting started
The script is written in Python, it can be run using any Python interpreter.

To calculate a set of probability density functions use the command: `pdfs = Pdf_from_quartiles(q1, q2, q3)`

Replace `q1`, `q2` and `q3` with the lower quartile, median and upper quartile respectively, e.g.: `pdfs = Pdf_from_quartiles(5, 9, 14)`

The constructor will automatically display a plot showing any distributions that have been successfully generated, together with some other information.

To see more detailed plots of a single distribution use:
* `pdfs.plot('skewnorm')`
* `pdfs.plot('gamma')`
* `pdfs.plot('weibull')`

To retrieve the _shape_ (`alpha`), _scale_ and _location_ parameters which define each distribution use:
* `pdfs.params('skewnorm')`
* `pdfs.params('gamma')`
* `pdfs.params('weibull')`

To retrieve some descriptive statistics for each distribution use:
* `pdfs.stats('skewnorm')`
* `pdfs.stats('gamma')`
* `pdfs.stats('weibull')`