from pytorch_fid import fid_score


def calc_fid(paths, batch_size=1, device="cuda", dims=2048):
    return fid_score.calculate_fid_given_paths(paths, batch_size, device, dims)
