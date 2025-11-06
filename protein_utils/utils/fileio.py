import io
import gzip

def file_handle_opener(filename):
    """ Figure out what opener to use based on filename.
        For now this function chooses between gzip and regular open
    """
    opener = open # default opener

    # if filename is actually a filehandle then pass it through
    def handle_passthrough(*args, **kwargs):
        return args[0]
    if isinstance(filename, io.IOBase): # filename is a filehandle actually
        opener = handle_passthrough
    else: # check if file is gzipped
        if str(filename).endswith(".gz"):
            opener = gzip.open
    return opener

def txt_io_gen(filename):
    """ Read in an MSA from a plain text file. One sequence per line. Can be
    gzipped or not. If gzipped, then the filename should end in .gz

    Return: an iteratory over the sequences in bytes 
    """
    opener = file_handle_opener(filename)
    with opener(filename, "rb") as fh: # open in bytes
        for line in fh:
            yield line.rstrip()


