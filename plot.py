import numpy as np
import matplotlib.pyplot as plt
con=np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
asr=np.array([0.16,0.24,0.24,0.36,0.40,0.44,0.44,0.48,0.56,1,1])
plt.figure()
plt.title('confidence-ASR')
plt.xlabel("confidence")
plt.ylabel("ASR")
plt.plot(con,asr)
plt.show()
