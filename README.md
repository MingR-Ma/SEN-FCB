# SEN-AFT
A Fast Symmertric and Topology-preserving Image Registration CNN Model with Two Sub-networks
## Train 
### Train SEN 
Use train_SEN.py `--train_path` [Your data path] to train with your own data. `./Data/ is default folder.
### Train AFT
Use `train_AFT.py` `--train_path` [Your data path] to train with your own data. `./Data/` is default folder.  
                 `--model_dir` [trained SEN weights or your trained weights] to get first model deformation fields to train AFT. Default './Model/weights/'  
## Test
Use `test.py` `--mode` [Swith to test 'SEN' or 'SEN+AFT'] `--fixed` [Image B] `--moving` [Image A]  
            `--model_dir` [trained SEN weights or your trained weights for test]
