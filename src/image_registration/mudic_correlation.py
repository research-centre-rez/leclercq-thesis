#!/home/erik/Documents/thesis/src/.venv3-11/bin/python
import sys
import muDIC as dic
import numpy as np

if __name__ == "__main__":
    print('loading .npy file')
    vid_mat = np.load('./temp.npy')

    image_stack = dic.image_stack_from_list(list(vid_mat))

    print('Created image stack')

    _, h, w = vid_mat.shape
    center_h     = h // 2
    center_w     = w // 2
    box_h, box_w = 100, 100
    num_elems_x  = 5
    num_elems_y  = 5

    offset_x = (box_w * num_elems_x) // 2
    offset_y = (box_h * num_elems_y)// 2
    print(f'Offset: {offset_x, offset_y}')

    upp_x = center_w - offset_x
    low_x = center_w + offset_x

    upp_y = center_h - offset_y #y=0 at the top of the image
    low_y = center_h + offset_y

    mesher = dic.Mesher()
    mesh   = mesher.mesh(images=image_stack,
                         Xc1=upp_x,
                         Xc2=low_x,
                         Yc1=upp_y,
                         Yc2=low_y,
                         n_elx=num_elems_x,
                         n_ely=num_elems_y,
                         GUI=False)


    print(mesh.Xc1, mesh.Yc1)
    print(mesh.Xc2, mesh.Yc2)
    
    ref_range  = 25
    ref_frames = list(np.arange(ref_range, len(vid_mat), ref_range))
    max_it     = 50
    
    print(ref_frames)

    inputs  = dic.DICInput(mesh=mesh,
                           image_stack=image_stack,
                           ref_update_frames=ref_frames,
                           maxit=max_it,
                           max_nr_im=len(vid_mat))

    dic_job = dic.DICAnalysis(inputs)
    results = dic_job.run()

    fields = dic.Fields(results)

    # The displacement is of shape [1, 2, i, j, n]
    # Where:
    # 1 because there is only one displacement matrix? 
    # 2 due to decomposing the displacement vectors into cartesian coords
    # i = number of grids in horizontal direction
    # j = number of grids in vertical direction
    # n = number of images (=length of vid_mat)
    displacement = fields.disp()

    np.save('displacement', displacement)

    #viz = dic.Visualizer(fields, images=image_stack)
    #viz.show(field="displacement", frame=3)
