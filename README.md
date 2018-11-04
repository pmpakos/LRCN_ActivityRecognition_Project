[Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389). 

Useful links :

* [Long-term Recurrent Convolutional Networks](http://jeffdonahue.com/lrcn/)

* [Caffe code for LRCN Activity Recognition](https://github.com/LisaAnne/lisa-caffe-public/tree/lstm_video_deploy/examples/LRCN_activity_recognition)

* [Sources Activity Recognition 1](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review)

* [Sources Activity Recognition 2](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5)

* [Sources Activity Recognition 3](https://github.com/cherrylawrence/learngit/tree/87e97be5c2b449a5ee61efc826008f45b8b4fcf0/Experiments/python%E5%AE%9E%E7%8E%B0/keras/ActionRecognition-master/ActionRecognition-master)

* [Intuitive understanding of 1D, 2D, and 3D Convolutions in Convolutional Neural Networks](https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n)

* [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

* [Convolution3D for Keras - Video Classification](https://github.com/axon-research/c3d-keras)

* [Using convolutional LSTM networks (or other models) to classify videos](https://github.com/sofiabroome/painface-recognition/blob/master/models.py)

* [ImageNet Models (Keras)](https://github.com/dandxy89/ImageModels)

* [Caffe prototxt Model Visualizer](http://ethereon.github.io/netscope/#/editor)

* [Jupyter-notebooks (Keras)](https://github.com/fchollet/deep-learning-with-python-notebooks)


---


1) To dataset που χρησιμοποιούμε είναι το [UCF-101](http://crcv.ucf.edu/data/UCF101.php). Κατέβασμα dataset ([frames & flow images](https://drive.google.com/drive/folders/0B_U4GvmpCOecMVIwS1lkSm5KTGM)) και εκτέλεση των scripts extraction_{frames,flow} 
Η χρήση flow images θα αποδειχθεί στη συνέχεια χρήσιμη γιατί μαθαίνει features που μέσω των rgb εικόνων δεν μπορούν να συλληφθούν. 


2) Tα αρχεία .py περιέχουν βοηθητικές συναρτήσεις που θα χρησιμοποιηθούν αργότερα στα .ipynb notebooks.

Στο **[custom_models.py](./custom_models.py)** ορίζονται τα δίκτυα που δοκίμασα, με τα πιο σημαντικά εξ αυτών να είναι το CaffeDonahueFunctional και το LSTMCaffeDonahueFunctional.

Στο **[image_processing.py](./image_processing.py)** υπάρχουν συναρτήσεις προεπεξεργασίας των frames και των flow_images.

Στο **[inception_utils.py](./inception_utils.py)** ορίζεται το δίκτυο που βασίζεται στο έτοιμο δίκτυο InceptionV3 του Keras, με freeze των κατάλληλων layers κάθε φορά. Χρησιμοποείται μόνο στο SingleFrameTraining_Inception_RGB.ipynb.

Στο **[LR_SGD.py](./LR_SGD.py)** ορίζεται ο ένας βελτιωμένος SGD optimizer, με τις αλλαγές που πρότεινε το δίκτυο του Caffenet.

Στο **[LRN2D.py](./LRN2D.py)** ορίζεται το Local Response Normalisation layer, το οποίο είχε αφαιρεθεί από τις τελευταίες εκδόσεις του Keras.

Στο **[utils.py](./utils.py)** ορίζονται κάποιες συναρτήσεις που φορτώνουν τα βάρη από το ένα δίκτυο στο άλλο, κατά τη μετάβαση από το single frame network στο lstm με τα time-distributed layers. Επίσης η συνάρτηση που αναλαμβάνει το compile του μοντέλου με τον custom SGD optimizer.

Στο **[video_processing.py](./video_processing.py)** ορίζονται οι συναρτήσεις προεπεξεργασίας των video για το LSTM network.

---

# Single Frame Network
Το συγκεκριμένο δίκτυο προέκυψε από το [CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet), με τροποποιήσεις στα layers. Αρχικά υπάρχουν 5 convolutional layers, σε συνδυασμό με max-pooling layers, ενώ στη συνέχεια 3 dense layers, με το απαραίτητο dropout ανάμεσα τους για αποφυγή overfitting. 


3) **Notebooks [SingleFrameTraining_Caffenet_RGB](SingleFrameTraining_Caffenet_RGB.ipynb) & [SingleFrameTraining_Caffenet_FLOW](SingleFrameTraining_Caffenet_FLOW.ipynb)**. Στα συγκεκριμένα γίνεται το training του δικτύου που παίρνει ξεχωριστά frames κάθε video, και εκπαιδεύεται να τα αναγνωρίζει. 
Λόγω του περιορισμένου μεγέθους στο dataset, κάθε frame crop-άρεται τυχαία, και προκύπτουν διαφορετικές "εκδοχές" κάθε τέτοιας εικόνας.

Το δίκτυο που προκύπτει θα χρησιμεύσει αργότερα σαν pretrained δίκτυο για το πιο πολύπλοκο LSTM network που θα αναλάβει να συλλάβει και χρονικές συσχετίσεις μεταξύ των frames.

4) **Notebook [SingleFrameTraining_Inception_RGB](SingleFrameTraining_Inception_RGB.ipynb)**. Εξετάζεται ένα διαφορετικό δίκτυο, που βασίζεται σε έτοιμη υλοποίηση του Keras, και σε single frame δίνει καλύτερα αποτελέσματα ταξινόμησης. Δεν υπάρχει υλοποίηση για FLOW, καθώς το training θα απαιτούσε πολύ περισσότερες επαναλήψεις από το RGB.

5) **Notebook [SingleFramePredictions](SingleFramePredictions.ipynb)**. Δοκιμές διαφορετικών συνδυασμών των frame και flow networks, δίνοντας μεγαλύτερη βαρύτητα σε ένα δίκτυο κάθε φορά. Χρησιμοποείται η τεχνική late fusion, δηλαδή κάθε δίκτυο δίνει δική του πρόβλεψη και στη συνέχεια γίνεται averaging των προβλέψεων. Παρατηρούμε ότι ο καλύτερος συνδυασμός δίνεται για 2/3 flow, 1/3 rgb προβλέψεις, δίνοντας μεγαλύτερη βαρύτητα δηλαδή στο τι προβλέψεις δίνονται από flow images. Για συνδυασμό με το inception rgb ο καλύτερος συνδυασμός είναι 1/2 flow, 1/2 rgb.

---

# LRCN Network
Το νέο δίκτυο εφαρμόζει ένα LSTM layer πάνω από το πρώτο dense layer του single frame network. Στο paper παρατήρησαν ότι δεν άξιζε να παραμείνει και το δεύτερο dense layer, καθώς δεν ανέβαζε αισθητά την ακρίβεια στο validation set. Όσο αυξάνεται το πλήθος των lstm units (από 256 έως 1024), τόσο καλύτερα αποτελέσματα δίνει, με πιο "βαρύ" δίκτυο όμως.

6) **Notebooks [VideoTraining_LSTM_RGB](VideoTraining_LSTM_RGB.ipynb) & [VideoTraining_LSTM_FLOW](VideoTraining_LSTM_FLOW.ipynb)**. Γίνεται εκπαίδευση του δικτύου αυτού με random 16-frames clips από το dataset, βοηθώντας έτσι και στο augmentation του dataset.

7) **Notebook [VideoTraining_LSTM_FLOW_1024](VideoTraining_LSTM_FLOW_1024.ipynb)**. Γίνεται εκπαίδευση του δικτύου για flow images με 1024 αντί για 512 lstm units.

8) **Notebook [VideoSequencePredictions](VideoSequencePredictions.ipynb)**. Λαμβάνονται 16-frame clips από κάθε video, με stride 8. Παίρνοντας το average των προβλέψεων για κάθε τέτοιο clip, λαμβάνουμε την συνολική πρόβλεψη για το video.

---

9) Στο **[app.py](./app.py)** γίνεται αξιολόγηση των διαφόρων ταξινομητών και συνδυασμών αυτών για ένα βίντεο ξεχωριστά.

---

Τα αποθηκευμένα βάρη των εκπαιδευμένων μοντέλων μπορούν να βρεθούν [εδώ](https://drive.google.com/drive/folders/1Eb_qrN7q7YJRAvzqDjXiVq-RRxp7Prhw).

0) pretrained_model : Τα βάρη που στο paper χρησιμοποιούν σαν initialization για το single frame network, pretrained στο ILSVRC-2012 dataset.

1) caffenet_single_rgb 

2) caffenet_single_flow

3) inception_rgb

4) lstm_rgb

5) lstm_flow

6) lstm_flow_1024