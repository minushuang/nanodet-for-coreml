# nanodet-for-coreml
this is a project about how to convert nanodet pytorch model to coreml and inference with it.<p> 
most code of this project are referenced from https://github.com/RangiLyu/nanodet
## requirements
to run this model, you need to install the pkgs in ```requierments.txt``` <p>
```
pip install -r requierments.txt
```
## how to use
run `export_coreml.py` to convert the pytorch model to coreml. <p>
run `demo_pytorch.py` to predict bboxes from images with pytorch model. <p>
run `demo_coreml.py` to predict bboxes from images with coreml model. <p>
the results are saved in worksapce/nanodet_m.<p>
the post process for pytorch model and coreml model is just the same.
### thanks
https://github.com/RangiLyu/nanodet
