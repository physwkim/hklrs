/* Golden cubes from binoculars-ng 7be4f79's own accumulation on the synthetic
 * K6C komega scan of bin_7be4f79_scan.h; the cubes are
 *   one_pass  hkl_binoculars_cube_new_empty, then cube_add_space per frame
 *   two_pass  hkl_binoculars_cube_new over all twenty spaces
 *   first_half, second_half  cube_add_space over frames 0-9 and 10-19
 *   merged    hkl_binoculars_cube_new_merge(first_half, second_half)
 * and each is dumped as its axes and its bins with contributions, by linear
 * offset. "coordinates" holds every pixel of every frame, masked or not,
 * at resolution 1e-12, for the bin-edge margin check.
 * Build and run: bin_7be4f79_cube.sh */
#include "bin_7be4f79_scan.h"

#define FINE 1e-12

static void row(const double *v, int n)
{
	printf("[");
	for (int j = 0; j < n; j++) printf("%s%.17g", j ? ", " : "", v[j]);
	printf("]");
}

static void cube(const char *key, const HklBinocularsCube *c, int first)
{
	HklBinocularsAxis *axis;
	size_t n = 1;
	int k = 0;
	printf("%s \"%s\": {\"axes\": [", first ? "" : ",\n", key);
	darray_foreach(axis, c->axes) {
		printf("%s{\"name\": \"%s\", \"resolution\": %.17g, \"imin\": %td, \"imax\": %td}",
		       k++ ? ", " : "", g_quark_to_string(axis->name), axis->resolution,
		       axis->imin, axis->imax);
		n *= axis->imax - axis->imin + 1;
	}
	printf("],\n  \"bins\": [");
	k = 0;
	for (size_t i = 0; i < n; i++)
		if (c->contributions[i])
			printf("%s[%zu, %.17g, %u]", k++ ? ", " : "", i, (double)c->photons[i],
			       c->contributions[i]);
	printf("]}");
}

int main(void)
{
	double pix[3 * N], fine[3] = {FINE, FINE, FINE};
	uint8_t mask[N];
	HklGeometry *g = scan_new(pix, mask);

	HklBinocularsSpace *spaces[FRAMES], *coords = hkl_binoculars_space_new(N, 3);
	HklBinocularsCube *one = hkl_binoculars_cube_new_empty(),
		*halves[2] = {hkl_binoculars_cube_new_empty(), hkl_binoculars_cube_new_empty()};
	ptrdiff_t at[FRAMES][N][3];
	int saved = quiet();
	for (int f = 0; f < FRAMES; f++) {
		scan_frame(g, f);
		spaces[f] = hkl_binoculars_space_new(N, 3);
		space(spaces[f], g, f, pix, resolutions, mask);
		hkl_binoculars_cube_add_space(one, spaces[f]);
		hkl_binoculars_cube_add_space(halves[f >= FRAMES / 2], spaces[f]);
		space(coords, g, f, pix, fine, NULL);
		for (int i = 0; i < N; i++)
			for (int c = 0; c < 3; c++)
				at[f][i][c] = darray_item(coords->items, i).indexes_0[c];
	}
	HklBinocularsCube *two = hkl_binoculars_cube_new(FRAMES, (const HklBinocularsSpace *const *)spaces);
	HklBinocularsCube *merged = hkl_binoculars_cube_new_merge(halves[0], halves[1]);
	loud(saved);

	printf("{\"binoculars_version\": \"7be4f79\", \"geometry\": \"K6C\",\n \"values_deg\": ");
	row(values0, 6);
	printf(", \"scanned\": \"komega\", \"step_deg\": %.17g, \"frames\": %d, \"wavelength\": %.17g,\n",
	       komega_step, FRAMES, wavelength);
	printf(" \"shape\": [%d, %d], \"origin\": ", SLOW, FAST);
	row(origin, 3);
	printf(", \"e_slow\": ");
	row(e_slow, 3);
	printf(", \"e_fast\": ");
	row(e_fast, 3);
	printf(",\n \"image\": \"(7 * pixel + 13 * frame) %% 17\", \"weight\": \"2^(frame %% 5 - 2)\", \"masked\": [");
	for (size_t m = 0; m < sizeof(masked_pixels) / sizeof(*masked_pixels); m++)
		printf("%s%d", m ? ", " : "", masked_pixels[m]);
	printf("],\n \"resolutions\": ");
	row(resolutions, 3);
	printf(", \"fine\": %.17g,\n \"coordinates\": [", FINE);
	for (int f = 0; f < FRAMES; f++) {
		printf("%s[", f ? ",\n  " : "");
		for (int i = 0; i < N; i++)
			printf("%s[%td, %td, %td]", i ? ", " : "", at[f][i][0], at[f][i][1], at[f][i][2]);
		printf("]");
	}
	printf("],\n \"cubes\": {\n");
	cube("one_pass", one, 1);
	cube("two_pass", two, 0);
	cube("first_half", halves[0], 0);
	cube("second_half", halves[1], 0);
	cube("merged", merged, 0);
	printf("}}\n");
	return 0;
}
