Όλο το project είναι βασισμένο στο paper [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389). 

Αλλα χρήσιμα links :

* http://jeffdonahue.com/lrcn/

* https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video

* https://github.com/LisaAnne/lisa-caffe-public/tree/lstm_video_deploy/examples/LRCN_activity_recognition


===


1) Κατέβασμα dataset [frames και flow images](https://drive.google.com/drive/folders/0B_U4GvmpCOecMVIwS1lkSm5KTGM) και εκτέλεση των scripts extraction_{frames,flow} 
Η χρήση flow images θα αποδειχθεί στη συνέχεια χρήσιμη γιατί μαθαίνει features που μέσω των rgb εικόνων δεν μπορούν να συλληφθούν.

Το [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset περιέχει 13320 video 101 κλάσεων 


2) Tα αρχεία .py περιέχουν βοηθητικές συναρτήσεις που θα χρησιμοποιηθούν αργότερα στα .ipynb notebooks.

Στο **[custom_models.py]()** ορίζονται τα δίκτυα που δοκίμασα, με τα πιο σημαντικά εξ αυτών να είναι το CaffeDonahueFunctional και το LSTMCaffeDonahueFunctional.

Στο **[image_processing.py]()** υπάρχουν συναρτήσεις προεπεξεργασίας των frames και των flow_images.

Στο **[inception_utils.py]()** ορίζεται το δίκτυο που βασίζεται στο έτοιμο δίκτυο InceptionV3 του Keras, με freeze των κατάλληλων layers κάθε φορά. Χρησιμοποείται μόνο στο SingleFrameTraining_Inception_RGB.ipynb.

Στο **[LR_SGD.py]()** ορίζεται ο ένας βελτιωμένος SGD optimizer, με τις αλλαγές που πρότεινε το δίκτυο του Caffenet.

Στο **[LRN2D.py]()** ορίζεται το Local Response Normalisation layer, το οποίο είχε αφαιρεθεί από τις τελευταίες εκδόσεις του Keras.

Στο **[utils.py]()** ορίζονται κάποιες συναρτήσεις που φορτώνουν τα βάρη από το ένα δίκτυο στο άλλο, κατά τη μετάβαση από το single frame network στο lstm με τα time-distributed layers. Επίσης η συνάρτηση που αναλαμβάνει το compile του μοντέλου με τον custom SGD optimizer.

Στο **[video_processing.py]()** ορίζονται οι συναρτήσεις προεπεξεργασίας των video για το LSTM network.

===

# Single Frame Network
Το συγκεκριμένο δίκτυο προέκυψε από το [CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet), με τροποποιήσεις στα layers. Αρχικά υπάρχουν 5 convolutional layers, σε συνδυασμό με max-pooling layers, ενώ στη συνέχεια 3 dense layers, με το απαραίτητο dropout ανάμεσα τους για αποφυγή overfitting. 


3) **Notebooks [SingleFrameTraining_Caffenet_RGB]() & [SingleFrameTraining_Caffenet_FLOW]()**. Στα συγκεκριμένα γίνεται το training του δικτύου που παίρνει ξεχωριστά frames κάθε video, και εκπαιδεύεται να τα αναγνωρίζει. 
Λόγω του περιορισμένου μεγέθους στο dataset, κάθε frame crop-άρεται τυχαία, και προκύπτουν διαφορετικές "εκδοχές" κάθε τέτοιας εικόνας.

Το δίκτυο που προκύπτει θα χρησιμεύσει αργότερα σαν pretrained δίκτυο για το πιο πολύπλοκο LSTM network που θα αναλάβει να συλλάβει και χρονικές συσχετίσεις μεταξύ των frames.

4) **Notebook [SingleFrameTraining_Inception_RGB]()**. Εξετάζεται ένα διαφορετικό δίκτυο, που βασίζεται σε έτοιμη υλοποίηση του Keras, και σε single frame δίνει καλύτερα αποτελέσματα ταξινόμησης. Δεν υπάρχει υλοποίηση για FLOW, καθώς το training θα απαιτούσε πολύ περισσότερες επαναλήψεις από το RGB.

5) **Notebook [SingleFramePredictions]()**. Δοκιμές διαφορετικών συνδυασμών των frame και flow networks, δίνοντας μεγαλύτερη βαρύτητα σε ένα δίκτυο κάθε φορά. Χρησιμοποείται η τεχνική late fusion, δηλαδή κάθε δίκτυο δίνει δική του πρόβλεψη και στη συνέχεια γίνεται averaging των προβλέψεων. Παρατηρούμε ότι ο καλύτερος συνδυασμός δίνεται για 2/3 flow, 1/3 rgb προβλέψεις, δίνοντας μεγαλύτερη βαρύτητα δηλαδή στο τι προβλέψεις δίνονται από flow images.

===

# LRCN Network
Το νέο δίκτυο εφαρμόζει ένα LSTM layer πάνω από το πρώτο dense layer του single frame network. Στο paper παρατήρησαν ότι δεν άξιζε να παραμείνει και το δεύτερο dense layer, καθώς δεν ανέβαζε αισθητά την ακρίβεια στο validation set. Όσο αυξάνεται το πλήθος των lstm units (από 256 έως 1024), τόσο καλύτερα αποτελέσματα δίνει, με πιο "βαρύ" δίκτυο όμως.

6) **Notebooks [VideoTraining_LSTM_RGB]() & [VideoTraining_LSTM_FLOW]()**. Γίνεται εκπαίδευση του δικτύου αυτού με random 16-frames clips από το dataset, βοηθώντας έτσι και στο augmentation του dataset.

7) **Notebook [VideoSequencePredictions]()**. Λαμβάνονται 16-frame clips από κάθε video, με stide 8. Παίρνοντας το average των προβλέψεων για κάθε τέτοιο clip, λαμβάνουμε την συνολική πρόβλεψη για το video. 

===

Τα αποθηκευμένα βάρη των εκπαιδευμένων μοντέλων μπορούν να βρεθούν [εδώ](https://drive.google.com/drive/folders/1Eb_qrN7q7YJRAvzqDjXiVq-RRxp7Prhw).
0) pretrained_model : Τα βάρη που στο paper χρησιμοποιούν σαν initialization για το single frame network, pretrained στο ILSVRC-2012 dataset.
1) caffenet_single_rgb 
2) caffenet_single_flow και caffenet_single_flow_extra. Στο extra, γίνεται για μερικές ακόμα επαναλήψεις training προκειμένου να πετύχει μείωσει του training loss.
3) inception_rgb
4) lstm_rgb_512
5) lstm_flow_512