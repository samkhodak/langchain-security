#include <fcntl.h>
void print_flag() {
    char flagbuf[256];
    int fd = open("flag.txt", O_RDONLY);
    read(fd, flagbuf, 256);
    write(1, flagbuf, 256);
    close(fd);
}
int main() {
    char buf[32];
    gets(buf);
    printf("Hi, %s\n", buf);
    return 0;
}