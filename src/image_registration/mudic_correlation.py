#!/home/erik/Documents/thesis/src/.venv3-11/bin/python
import sys
import muDIC as dic
import numpy as np
import argparse
import video_matrix

parser = argparse.ArgumentParser(description='Estimating correlation between individual frames of the video matrix')

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

# Required arguments
required.add_argument('-i', '--input', type=str, required=True, help='Path to the input video, can be .npy file or .mp4')

# Optional arguments
optional = parser.add_argument_group('optional arguments')
optional.add_argument('-o', '--save_as', type=str, help='Name of the output file')
optional.add_argument('--box_h', type=int, default=100, help='Height of the correlation cell')
optional.add_argument('--box_w', type=int, default=100, help='Width of the correlation cell')
optional.add_argument('--num_elems_x', type=int, default=5, help='How many cells in the x axis')
optional.add_argument('--num_elems_y', type=int, default=5, help='How many cells in the y axis')
optional.add_argument('--max_it', type=int, default=50, help='Max number of iterations in the correlation step')
optional.add_argument('--ref_range', type=int, default=25, help='How often should the ref frame be updated')
parser._action_groups.append(optional)

def create_mesh(h,w, image_stack, arguments):
    center_h     = h // 2
    center_w     = w // 2

    offset_x = (arguments.box_w * arguments.num_elems_x) // 2
    offset_y = (arguments.box_h * arguments.num_elems_y) // 2

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
                         n_elx=arguments.num_elems_x,
                         n_ely=arguments.num_elems_y,
                         GUI=False)
    return mesh

def main(args):
    print('Running with the following parameters:')
    for arg in vars(args):
        print(f'  {arg}: {getattr(args, arg)}')

    base_name, file_ext = args.input.split('/')[-1].split('.')

    if file_ext == 'mp4':
        print('Processing .mp4 video')
        out = video_matrix.create_video_matrix(args.input)
        print('Rotating video frames')
        vid_mat = video_matrix.rotate_frames(out, save_as=None)
    else:
        try:
            print('Loading .npy file')
            vid_mat = np.load(args.input)
        except OSError as e:
            print('Could not load the .npy file, please try again')
            print(e)
            sys.exit(-1)

    image_stack = dic.image_stack_from_list(list(vid_mat[15:]))

    print('Image stack successfully created')

    _, h, w = vid_mat.shape
    mesh = create_mesh(h, w, image_stack, args)

    ref_frames = list(np.arange(args.ref_range, len(vid_mat), args.ref_range))

    print(f'Reference frame update will happen at these frames:\n  {ref_frames}')

    inputs  = dic.DICInput(mesh=mesh,
                           image_stack=image_stack,
                           ref_update_frames=ref_frames,
                           maxit=args.max_it,
                           noconvergence='ignore',
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

    if args.save_as is None:
        np.save('displacement', displacement)
    else:
        np.save(args.save_as, displacement)

    # For visualisation purposes
    #viz = dic.Visualizer(fields, images=image_stack)
    #viz.show(field="displacement", frame=3)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
