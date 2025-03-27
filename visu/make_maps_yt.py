import yt
import numpy as np
import matplotlib.pyplot as plt
import os

# sim_dir = "/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/id2757_995pcnt_mstel/"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"
snap = 294

out_dir = os.path.join(sim_dir, f"output_{snap:05d}", f"info_{snap:05d}.txt")

ds = yt.load(out_dir, fields=["density", "temperature"])


d_sl = yt.SlicePlot(
    ds, "z", "temperature", center=("max", "density"), width=(20, "kpc")
)
d_sl.save("test.png")

# # d_pr = yt.ProjectionPlot(
# #     ds, "z", "density", center=("max", "density"), width=(20, "kpc")
# # )
# # d_pr.save("test_proj.png")

# # d_pr = yt.ProjectionPlot(
# #     ds,
# #     "z",
# #     "temperature",
# #     center=("max", "density"),
# #     width=(20, "kpc"),
# #     weight_field="density",
# # )
# # d_pr.set_zlim("temperature", 0)
# # d_pr.save("test_proj_temp.png")


# ctr = np.asarray([0.248782, 0.211555, 0.318053])
# r = 0.001
# bbox = [ctr - r, ctr + r]
# ds = yt.load(
#     out_dir, bbox=bbox
# )  # bbox returns a subset of the data - might not be cubic due to hilbert decomp
# box = ds.box(
#     left_edge=bbox[0], right_edge=bbox[1]
# )  # box cuts off the data outside of bbox
# # load a portion of the data
# data = box["Density"]
# data.convert_to_units("g/cm**3")
# print(data.max())

# # # print(box[("io", "star_age")].min(), box[("io", "star_age")].max())
# # # io -> stars/debris; sink; nbody -> dark matter
# # # negative ages are not stars


# # Create a 150 kpc radius sphere, centered ctr
# sp = ds.sphere(ctr, (150, "kpc"))

# # Use the total_quantity derived quantity to sum up the
# # values of the mass and particle_mass fields
# # within the sphere.
# particle_mass = sp.quantities.total_quantity([("gas", "mass"), ("io", "particle_mass")])

# stars = sp[("io", "star_age")] > 0
# stellar_mass = sp[("io", "particle_mass")][stars].sum()
# stellar_mass.convert_to_units("msun")

# print("Total mass of stars within 150 kpc of the center:", stellar_mass)


# # ts = yt.load(os.path.join(sim_dir, "output_*"))
