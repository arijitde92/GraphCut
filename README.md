# GraphCut
Python 3 implementation of GraphCut

## Credits
Mostly reimplemented from [https://github.com/julie-jiang/image-segmentation/](https://github.com/julie-jiang/image-segmentation/).<br>
Please visit [https://julie-jiang.github.io/image-segmentation/](https://julie-jiang.github.io/image-segmentation/) for theory on Graph cuts.

## Usage
``` 
python graphcut.py filename
```
Run
<br>
```
python graphcut.py -h
```
for information on other command line arguments and to choose between the type of optimization technique: (Augmenting Path or Push Relabel)
<br>
## How to seed
Once the program runs, a new window will pop up showing your image. Use your cursor to mark object seeds, which would be shown in red. Once you're done, press `esc`.

Then do the same to mark background seeds, which would be shown in green.
Then again press `esc` and wait for the computation to finish.

## Dependencies
- numpy
```
pip install numpy
```
- opencv-python
```
pip install opencv-python
```
