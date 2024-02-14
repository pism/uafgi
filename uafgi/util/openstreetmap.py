import io,os
import urllib.request
import PIL
import cartopy.io.img_tiles

class CartopyOSM(cartopy.io.img_tiles.OSM):
    """
    CartopyOSM(..., cache=<cache-dir>)
    """

    # https://www.theurbanist.com.au/2021/03/plotting-openstreetmap-images-with-cartopy/
    #@override
    def get_image(self, tile):
        """This function reformats web requests from OSM for cartopy
        Heavily based on code by Joshua Hrisko at:
            https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
        """

        leaf = 'osmtile-{}.png'.format('-'.join(str(x) for x in tile))
        img_file = self.cache_path / leaf

        if os.path.exists(img_file):
            print(f'OSM Reading existing tile: {img_file}')
            img = PIL.Image.open(img_file)
        else:
            print(f'OSM Writing new tile: {img_file}')
            url = self._image_url(tile)                # get the url of the street map API
            req = urllib.request.Request(url)                         # start request
            req.add_header('User-agent','Anaconda 3')  # add user agent to request
            fh = urllib.request.urlopen(req)

            data = fh.read()
            fh.close()                                 # close url

            # Store it away
            os.makedirs(img_file.parents[0], exist_ok=True)
            with open(img_file, 'wb') as out:
                out.write(data)

            im_data = io.BytesIO(data)            # get image
            img = PIL.Image.open(im_data)                  # open image with PIL

        img = img.convert(self.desired_tile_form)  # set image format
        return img, self.tileextent(tile), 'lower' # reformat for cartopy


def plot_layer(ax, scale, cache=False, **kwargs):
    """
    scale:
        Scale specifications should be selected based on radius
        but be careful not have both large scale (>16) and large radius (>1000), 
         it is forbidden under [OSM policies](https://operations.osmfoundation.org/policies/tiles/)
        -- 2     = coarse image, select for worldwide or continental scales
        -- 4-6   = medium coarseness, select for countries and larger states
        -- 6-10  = medium fineness, select for smaller states, regions, and cities
        -- 10-12 = fine image, select for city boundaries and zip codes
        -- 14+   = extremely fine image, select for roads, blocks, buildings
    cache:
        Directory where downloaded OSM files are cached.
    """

#    cartopy.io.img_tiles.OSM.get_image = image_spoof # reformat web request for street map spoofing
    img = CartopyOSM(cache=cache)
    #stroke = [pe.Stroke(linewidth=1, foreground='w'), pe.Normal()]

    # or change scale manually
    return ax.add_image(img, scale, **kwargs) # add OSM with zoom specification
