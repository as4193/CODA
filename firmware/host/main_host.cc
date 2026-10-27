// Virtual CODA board for the Mac: the same firmware/common code and TFLM tree (CMSIS-NN
// kernels in their portable C path), talking the serial protocol over a pseudo-terminal.
// Lets scripts/board_probe.py be tested end to end without hardware:
//   make -C firmware/host && firmware/host/coda_host     (prints the port to use)
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <util.h>  // openpty (macOS); use <pty.h> on Linux

#include "coda_board.h"
#include "coda_runtime.h"

static int g_fd = -1;

extern "C" {
int board_read(uint8_t* buf, int max) {
  int r = (int)read(g_fd, buf, max);
  if (r <= 0) { usleep(50); return 0; }
  return r;
}
void board_write(const uint8_t* buf, int n) {
  while (n > 0) {
    int w = (int)write(g_fd, buf, n);
    if (w > 0) { buf += w; n -= w; } else usleep(50);
  }
}
static uint64_t ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
uint32_t board_millis(void) { return (uint32_t)(ns() / 1000000ull); }
uint32_t board_cycles(void) { return (uint32_t)ns(); }  // "cycles" = nanoseconds here
uint32_t board_cpu_hz(void) { return 1000000000u; }
const char* board_name(void) { return "host-sim"; }
}

int main() {
  int master, slave;
  char name[128];
  if (openpty(&master, &slave, name, nullptr, nullptr) != 0) { perror("openpty"); return 1; }
  struct termios tio;
  tcgetattr(slave, &tio);
  cfmakeraw(&tio);
  tcsetattr(slave, TCSANOW, &tio);
  g_fd = master;
  fcntl(g_fd, F_SETFL, fcntl(g_fd, F_GETFL) | O_NONBLOCK);
  coda_setup();
  printf("%s\n", name);
  fflush(stdout);
  for (;;) coda_poll();
}
