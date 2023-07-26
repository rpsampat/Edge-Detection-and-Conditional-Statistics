# Edge-Detection-and-Conditional-Statistics
Detects edge of a free shear flow field such as a jet in coflow from PIV data. Conditional statistics in the frame of reference of the interface are calculated with other higher order terms and derivates such as kinetic energy budget terms, enstrophy and enstrophy flux

The code can be executed through Run.py. DataProcessingConditonal.py assimilates all conditional data. The Edge detection from instantaneous images is implemented in EdgeDetect.py. For each image, InterfaceDetection.py is execued. This is where the quantitties are sampled at conditional coordinates.

