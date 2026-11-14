/* An LD_PRELOAD shim for a controller on a host whose timer is coarse: sleeps
 * that end on time.
 *
 * OrbStack's VM (kernel 7.2.12, 2026-11-14) wakes a sleeper on a 3 ms tick:
 * sleep(0.1 ms .. 2 ms) takes 3.0 ms, 4 and 5 ms take 6. Timer slack of
 * 1 ns, select(), SCHED_FIFO and other cores kept busy change nothing; only
 * spinning is on time. The robot's controller paces its 1 kHz loop with a
 * 1 ms sleep and counts time in ticks, so there it ran at 334 Hz with its
 * clock at a third of the wall's: stand-up took 5.5 s for 2, the 50 Hz policy
 * ran at 17 and walked 0.5 m in 8 s. With this loaded: 997-999 Hz, 2.1 s,
 * 2.67 m, for about 1.3 cores (its loop, IMU and BMS threads spin).
 *
 * nanosleep / clock_nanosleep: what is longer than TICK_NS is slept, the rest
 * is spun. Built by scripts/spin_sleep.sh into run/, loaded through
 * docker-compose.controller.yml (G2_SPIN_SLEEP). Not for the robot or a
 * Linux host, whose sleeps are on time.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <time.h>
/* The tick and a margin: a sleep of (left - TICK_NS) then never ends late. */
#define TICK_NS 3500000LL
static long long now_ns(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
static void until(clockid_t c, long long end) {
    static int (*real)(clockid_t, int, const struct timespec *, struct timespec *);
    if (!real) real = dlsym(RTLD_NEXT, "clock_nanosleep");
    long long left = end - now_ns(c);
    if (left > TICK_NS) { long long s = left - TICK_NS; struct timespec t = { s / 1000000000LL, s % 1000000000LL }; real(CLOCK_MONOTONIC, 0, &t, 0); }
    while (now_ns(c) < end) ;
}
int clock_nanosleep(clockid_t c, int flags, const struct timespec *req, struct timespec *rem) {
    if (!req || req->tv_nsec < 0 || req->tv_nsec >= 1000000000L) return EINVAL;
    long long r = req->tv_sec * 1000000000LL + req->tv_nsec;
    until(c, (flags & TIMER_ABSTIME) ? r : now_ns(c) + r);
    return 0;
}
int nanosleep(const struct timespec *req, struct timespec *rem) {
    int e = clock_nanosleep(CLOCK_MONOTONIC, 0, req, rem);
    if (e) { errno = e; return -1; }
    return 0;
}
