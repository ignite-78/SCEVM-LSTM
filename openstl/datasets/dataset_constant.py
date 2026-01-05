dataset_parameters = {
    'cikm': {
        'in_shape': [5, 1, 128, 128],
        'pre_seq_length': 5,
        'aft_seq_length': 10,
        'total_length': 15,
        'data_name': 'cikm',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
}