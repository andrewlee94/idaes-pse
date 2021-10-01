Visualizing Surrogate Model Results
===================================

The *idaes.surrogate.plotting.sm_plotter* module contains a number of methods to assess the accuracy
and performance of a surrogate model fit. Once a surrogate model is trained by an external method
(e.g. *idaes.surrogate.alamopy.AlamoTrainer*), the present methods automatically plot and format data
passed as arguments. Note that these methods, although optimized for surrogate model integration, are
solely visualization tools and any passed data such as residual error values must be calculated
beforehand.

*idaes.surrogate.plotting.sm_plotter* contains six utility methods for plotting and formatting data:

* *sm_plotter.scatter2D*, *sm_plotter.scatter3D*, *sm_plotter.parity*, *sm_plotter.residual* - plotting methods to visualize passed data

* *sm_plotter.pdfPrint*, *sm_plotter.extractData* - auxilliary methods to format data and plots

Dependencies
------------
In addition to built-in Python functions, the visualization methods depend on the following packages being installed (imported inside module):

* *numpy*

* *matplotlib.pyplot*

* *matplotlib.backends.backend_pdf*

* *itertools.combinations*

Basic Usage
-----------
To use the packages, they must be imported from *idaes.surrogate.plotting.sm_plotter* and the arguments to pass must be defined:

.. code:: python

   # Required imports
   >>> import numpy as np
   >>> import pandas as pd
   >>> from idaes.surrogate.plotting import sm_plotter as splot
   >>> import pandas as pd

   # Load dataset from a csv file and train the model
   >>> xy_data = pd.read_csv('data.csv', header=None, index_col=0)
   >>> xdata = data.iloc[:, :numins]  # separate out the input vars from the dataset
   >>> xlabels = xdata.columns
   >>> xtest = [sampling_method](xdata)  # if desired, can partition data into training and test sets
   >>> zdata = data.iloc[:, numins:]  # and the rest of the data are output vars
   >>> zlabels = zdata.columns
   >>> zfit = [training_method](xdata, zdata, **options)  # visualization methods are independent of surrogate training method
   
   # Prepare arguments for plotting methods
   >>> xdata = splot.extractData(xdata)  # converts data to 2D numpy array (to prevent indexing errors while plotting)
   >>> zdata = splot.extractData(zdata)
   >>> xtest = splot.extractData(xtest)
   >>> zfit = splot.extractData(zfit)
   >>> e = zfit - zdata
   
   # Call visualization methods on datasets
   >>> splot.scatter2D(xdata, zdata, xtest, zfit, **kwargs)
   >>> splot.scatter3D(xdata, zdata, xtest, zfit, **kwargs)
   >>> splot.parity(zdata, zfit, **kwargs)
   >>> splot.residual(xdata, e, **kwargs)

* **xdata** is a two-dimensional numpy array containing input variable training data; originates from external data file or input to sampling function
* **zdata** is a two-dimensional numpy array containing output variable training data; originates from external data file or sampled distribution
* **xtest** is a two-dimensional numpy array containing partitioned input variable data; a subset of rows of xdata or samples from the same domain, with the same number of columns
* **zfit** is a two-dimensional numpy array containing regressed output variable values from the surrogate training method; same shape as zdata
* **e** is a two-dimensional numpy array containing error values of interest; required for *sm_plotter.residual* only and must be be calculated prior to calling the method

**Optional Arguments**

* **xlabels** - list of strings containing input variable names, which can be specified as above but will be generated automatically as needed; defaults to ('x1', 'x2', ...)
* **zlabels** - list of strings containing output variable names, which can be specified as above but will be generated automatically as needed; defaults to ('z1', 'z2', ...)
* **elabels** - list of strings containing error names for *sm_plotter.residual* method, which may be specified if desired; defaults to ('z1 Error', 'z2 Error', ...)
* **show** - logical argument which determines if plot windows will be shown; defaults to True
* **PDF** - logical argument which determines if plots are exported to a PDF document; defaults to False, if set to True will call *sm_plotter.pdfPrint(fig, filename)* with required arguments **fig** containing figure() objects and associated axes (internally defined), and filename to call PDF file (see below)
* **filename** - string expression which must be of the form '.pdf' (e.g. filename='example.pdf', filename='alamo_parity.pdf') used as input for *sm_plotter.pdfPring*; defaults are set based on the plotting method itself, for example *sm_plotter.scatter2D* defaults to filename='results_scatter2D.pdf'. Note that this will overwrite any previous files with the same names, and specifying unique filename arguments is recommended for plotting/printing results from multiple surrogate models.


Surrogate Training Integration
------------------------------

The result of the surrogate training process can be passed directly to the visualization tools in a systematic fashion.
The following code snippet demonstrates how the visualization tools may be integrated with other surrogate tools:

.. code:: python

   # Required imports
   >>> import numpy as np
   >>> import pandas as pd
   >>> import sm_plotter as splot
   >>> from idaes.surrogate.alamopy_new import AlamoTrainer, AlamoObject
   >>> from idaes.surrogate.pysmo import sampling as sp

   # Import Auto-reformer training data and generate test data samples
   >>> np.set_printoptions(precision=6, suppress=True)

   >>> data = pd.read_csv(r'reformer-data.csv')
   >>> train_data = data.iloc[::28, :]

   >>> xdata = train_data.iloc[:, :2]
   >>> zdata = train_data.iloc[:, 2:]
   >>> xlabels = xdata.columns
   >>> zlabels = zdata.columns

    # Generate test samples for validation
   >>> bounds_min = xdata.min(axis=0)
   >>> bounds_max = xdata.max(axis=0)
   >>> bounds_list = [list(bounds_min), list(bounds_max)]
   >>> space_init = sp.LatinHypercubeSampling(bounds_list,
   >>>                                        sampling_type='creation',
   >>>                                        number_of_samples=100)
   >>> xtest = np.array(space_init.sample_points())

   # Call AlamoTrainer to generate surrogate model fit
   >>> x, z = np.array(xdata), np.array(zdata)
   >>> trainer = AlamoTrainer()
   >>> trainer._n_inputs = np.shape(x)[0]
   >>> trainer._n_outputs = np.shape(z)[0]
   >>> trainer._rdata_in = x
   >>> trainer._rdata_out = z

   # data bounds and labels
   >>> xmin, xmax = [0.1, 0.8], [0.8, 1.2]
   >>> trainer._input_min = [xmin[i] for i in range(len(xmin))]
   >>> trainer._input_max = [xmax[i] for i in range(len(xmax))]
   >>> trainer._input_labels = [xlabels[i] for i in range(len(xlabels))]
   >>> trainer._output_labels = [zlabels[i] for i in range(len(zlabels))]

   # surrogate training options

   >>> aoptlabels = ['constant', 'linfcns', 'multi2power', 'monomialpower',
   >>>               'ratiopower', 'maxterms', 'filename', 'overwrite_files']
   >>> aoptvals = [True, True, (1, 2), (2, 3, 4, 5, 6), (1, 2), [10] * len(zlabels),
   >>>             'alamo_run', True]
   >>> options = dict(zip(aoptlabels, aoptvals))
   >>> for entry in options:
   >>>     setattr(trainer.config, entry, options[entry])

   # Train surrogate and generate predicted values
   >>> trainer.train_surrogate()
   >>> surrogate = trainer._results['Model']
   >>> input_labels = trainer._input_labels
   >>> output_labels = trainer._output_labels
   >>> input_bounds = {xlabels[i]: (xmin[i], xmax[i])
   >>>                 for i in range(len(xlabels))}

   >>> alm_surr = AlamoObject(surrogate, input_labels,
   >>>                        output_labels, input_bounds)
   >>> zfit = alm_surr.evaluate_surrogate(xtest)

   # Call visualization tools on surrogate model results
   >>> xdata = splot.extractData(xdata)
   >>> xtest = splot.extractData(xtest)
   >>> zdata = splot.extractData(zdata)
   >>> zfit = splot.extractData(zfit)

   >>> splot.parity(zdata, zfit, zlabels=zlabels, PDF=True,
   >>>              filename=('alamo_example_parity.pdf'))

An example of a plot that is produced:

.. image:: /images/surr-plotting-example.png
    :width: 600px
    :align: center

Available Methods
------------------

.. automethod:: idaes.surrogate.plotting.sm_plotter.scatter2D
.. automethod:: idaes.surrogate.plotting.sm_plotter.scatter3D
.. automethod:: idaes.surrogate.plotting.sm_plotter.parity
.. automethod:: idaes.surrogate.plotting.sm_plotter.residual
.. automethod:: idaes.surrogate.plotting.sm_plotter.pdfPrint
.. automethod:: idaes.surrogate.plotting.sm_plotter.extractData
