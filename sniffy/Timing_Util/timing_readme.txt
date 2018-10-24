This is the version of the Sniffy UMA script used for generating the BUA performance data in the technical report.

The Python script itself ('sniffy_timing.py') executes a signle run of Sniffy. Run it without input parameters to get a list of required input parameters.
The experiment described in the technical report was executed by running the batch script 'sniffy_timing.bat', with the batch script 'sniffy_timing_single.bat' controlling the parameters for a single run.

'batchA.dat' and 'batchB.dat' are the pickled output files of the experimental run reported in the technical report.

Plotting was done using the matpolotlib library, by running 'sniffy_timing_plot.py'.

