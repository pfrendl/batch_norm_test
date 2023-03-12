# batch_norm_test
This repository experimentally demonstrates on the CIFAR10 dataset that while batch normalization helps with classification, it can significantly hurt autoencoding performance. You can read more on this topic [here](https://pfrendl.substack.com/).

Results for classification:

![image](https://user-images.githubusercontent.com/6968154/224574571-c0f28982-f340-411a-8837-f69260537cce.png)

Results for autoencoding:

![image](https://user-images.githubusercontent.com/6968154/224574594-2b2614d1-f272-4d87-ae71-b4a8d1fc2a68.png)


Run the trainings:

```
python classifier.py
python autoencoder.py
```

Track the trainings through the browser:

```
python dashboard.py
```

Generate and save figures of the logged trainings:

```
python plotting.py
```
