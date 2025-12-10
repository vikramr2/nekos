#ifndef MAPPED_FILE_H
#define MAPPED_FILE_H

#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Memory-mapped file reader
class MappedFile {
private:
    void* mapped_data = nullptr;
    size_t file_size = 0;
    int fd = -1;

public:
    ~MappedFile() {
        if (mapped_data) munmap(mapped_data, file_size);
        if (fd >= 0) close(fd);
    }
    
    bool open(const std::string& filename) {
        fd = ::open(filename.c_str(), O_RDONLY);
        if (fd == -1) return false;
        
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            fd = -1;
            return false;
        }
        
        file_size = sb.st_size;
        mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        
        if (mapped_data == MAP_FAILED) {
            close(fd);
            fd = -1;
            mapped_data = nullptr;
            return false;
        }
        
        // Advise the kernel that we'll access this sequentially
        madvise(mapped_data, file_size, MADV_SEQUENTIAL);
        return true;
    }
    
    const char* data() const { return static_cast<const char*>(mapped_data); }
    size_t size() const { return file_size; }
};

#endif // MAPPED_FILE_H
