class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return './VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif database == 'davis':
            return './DAVIS'
        elif database == 'sbd':
            return './benchmark_RELEASE/' # folder that contains dataset/.
        elif database == 'cityscapes':
            return '/path/to/Segmentation/cityscapes/'         # foler that contains leftImg8bit/
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
