# file has a different naming convention because python cannot 
# import modules with "-" in the file name
import eda_h5_melissa as edah5
import matplotlib.pyplot as plt

sparse = edah5.csr_matrix


plt.spy(sparse.toarray())
#plt.show()
plt.savefig('analysis/eda/h5-plots/2d-plot.png')

