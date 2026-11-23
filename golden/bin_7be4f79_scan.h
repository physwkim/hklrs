/* The synthetic K6C komega scan of bin_7be4f79_cube.c and bin_7be4f79_hdf5.c:
 * 20 frames, a 5x6 flat detector, an integer image, one weight per frame and
 * two masked pixels; each frame is a binoculars-ng 7be4f79 qcustom QX_QY_QZ
 * space. The weights are powers of two, so image * weight and its float sums
 * are exact. */
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include "hkl.h"
#include "hkl-binoculars.h"
#include "hkl-binoculars-private.h"

#define FRAMES 20
#define SLOW 5
#define FAST 6
#define N (SLOW * FAST)

/* mu, komega, kappa, kphi, gamma, delta in degrees; komega is scanned */
static const double values0[] = {3, 10, -30, 40, 12, 25};
static const double komega_step = 0.5;
static const double wavelength = 1.54;
static const double origin[] = {0.5, -0.00375, -0.003}, e_slow[] = {0, 0, 0.0015},
	e_fast[] = {0, 0.0015, 0};					/* m */
static const double resolutions[] = {0.0214, 0.0177, 0.0199};
static const int masked_pixels[] = {7, 22};

static double image(int f, int i) { return (7 * i + 13 * f) % 17; }
static double weight(int f) { return ldexp(1, f % 5 - 2); }

/* qcustom prints its matrices to stdout when built without NDEBUG
 * (cglm/io.h) */
static int quiet(void)
{
	fflush(stdout);
	int saved = dup(1), null = open("/dev/null", O_WRONLY);
	dup2(null, 1);
	close(null);
	return saved;
}

static void loud(int saved)
{
	fflush(stdout);
	dup2(saved, 1);
	close(saved);
}

/* the K6C geometry at the scan's wavelength; fills the pixel positions
 * (one plane per coordinate) and the mask */
static HklGeometry *scan_new(double *pix, uint8_t *mask)
{
	HklGeometry *g = hkl_factory_create_new_geometry(hkl_factory_get_by_name("K6C", NULL));
	hkl_geometry_wavelength_set(g, wavelength, HKL_UNIT_USER, NULL);
	for (int i = 0; i < SLOW; i++)
		for (int j = 0; j < FAST; j++)
			for (int c = 0; c < 3; c++)
				pix[c * N + i * FAST + j] = origin[c] + i * e_slow[c] + j * e_fast[c];
	for (int i = 0; i < N; i++) mask[i] = 0;
	for (size_t m = 0; m < sizeof(masked_pixels) / sizeof(*masked_pixels); m++)
		mask[masked_pixels[m]] = 1;
	return g;
}

static void scan_frame(HklGeometry *g, int f)
{
	double values[6];
	for (int a = 0; a < 6; a++) values[a] = values0[a];
	values[1] += f * komega_step;
	hkl_geometry_axis_values_set(g, values, 6, HKL_UNIT_USER, NULL);
}

static void space(HklBinocularsSpace *s, HklGeometry *g, int f, const double *pix,
		  const double *res, const uint8_t *mask)
{
	double img[N], flat[N];
	size_t dims[3] = {3, SLOW, FAST};
	for (int i = 0; i < N; i++) {
		img[i] = image(f, i);
		flat[i] = 1;
	}
	hkl_binoculars_space_qcustom_double(s, g, img, N, weight(f), flat, (double *)pix, 3, dims,
		(double *)res, 3, mask, HKL_BINOCULARS_SURFACE_ORIENTATION_HORIZONTAL, NULL, 0, 0, 0, 0,
		HKL_BINOCULARS_QCUSTOM_SUB_PROJECTION_QX_QY_QZ, 0, 0, 0, "komega", false, NULL);
}
