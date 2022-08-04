# file has a different naming convention because python cannot 
# import modules with "-" in the file name
import eda_h5_melissa as edah5
import matplotlib.pyplot as plt

sparse = edah5.csr_matrix

plt.spy(csr_matrix.toarray())
plt.show()

#plt.hist

