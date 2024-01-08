import os
import numpy as np
import tifffile
from valis import registration


def run_valis(
    src_dir,
    res_dir, 
    ref_slide=None,
    micro=False,
    kill_jvm=False,
    **kwargs
):    
    """
    Registration w/ VALIS
    Reference: https://www.nature.com/articles/s41467-023-40218-9
    """
    # Additional argument settings
    args = {
        'img_list': None,                            # Specify for aligning subset of imgs.
        'series': 0,                                 # Resolution series # for pyramid formatted imgs
        'align_to_ref': True,                        # Aligning `to` vs. `towards` the ref. image
        'image_type': 'brightfield',                 # Registration image type BF / Fluorescence
        'micro_res': 2000,                           # Resolution for valis micro-registration
        'warped_fname': 'valis_stacked.ome.tif'      # Warped stacked output filename
    }

    for k, v in kwargs.items():
        args[k] = v

    if ref_slide is not None:
        registrar = registration.Valis(src_dir,
                                       res_dir, 
                                       series=args['series'],
                                       # img_list=args['img_list'],
                                       reference_img_f=ref_slide, 
                                       align_to_reference=args['align_to_ref'], 
                                       imgs_ordered=True,
                                       image_type=args['image_type'])
        
    else:
        registrar = registration.Valis(src_dir, 
                                       res_dir, 
                                       imgs_ordered=True)
        
    rigid_registrar, non_rigid_registrar, _ = registrar.register()

    if micro:
        registrar.register_micro(max_non_rigid_registration_dim_px=args['micro_res'], align_to_reference=True)

    # save results
    save_dir = os.path.join(res_dir, "registered_slides")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    registrar.warp_and_save_slides(save_dir, crop="overlap")

    if kill_jvm:
        registration.kill_jvm()
    else:
        print("NOTE: JVM HAS NOT BEEN KILLED. Make sure to run kill_jvm() at the end of your script.")

    # aligned_imgs = [tifffile.imread(os.path.join(save_dir, f))
    #                 for f in sorted(os.listdir(save_dir))
    #                 if f[-8:] == 'ome.tiff']

    # aligned_imgs = np.array(aligned_imgs)
    # aligned_imgs = aligned_imgs.transpose((3,0,1,2))
    # tifffile.imwrite(os.path.join(save_dir, 'valis_stacked.ome.tif'), aligned_imgs, metadata={'axes': 'CZYX'})

    # print("Aligned stacked image saved to:", os.path.join(save_dir, args['warped_fname']))

    return registrar, rigid_registrar, non_rigid_registrar


def kill_jvm():
    registration.kill_jvm()
