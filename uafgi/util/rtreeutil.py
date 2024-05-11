import rtree
import shapely
#import osgeo.ogr

class RTree:
    def __init__(self, df, shapecol='shape'):
        """
        df:
            Dataframe made via ogrutil (NOT shputil)
            Index must be an integer, and unique
        """

        self.df = df
        self.shapecol = shapecol

        # Make the rtree
        self.idx = rtree.index.Index(interleaved=True)
        for ix,row in df.iterrows():    # (index, Series) pair
            geom = row[shapecol]

            # OGR code: deprecated due to OGR memory leak
            #env = geom.GetEnvelope()    # (minX, maxX, minY, maxY)
            #bbox = (env[0], env[2], env[1], env[3])    # Interleaved: minX, minY, maxX, maxY
            #print('ix ', ix)

            # Shapely code
            # https://stackoverflow.com/questions/20094346/is-there-an-envelope-class-in-shapely

            bbox = geom.bounds    # Interleaved: minX, minY, maxX, maxY
            self.idx.insert(ix, bbox)

    def intersection(self, geom):
        """Returns index values of items intersecting with geom
        geom:
            Shapely Geometry
        """

        # Do a rough cut...
        subrows = list()    # Index of rows we want to keep
        # OGR
        #env = geom.GetEnvelope()
        #bbox = (env[0], env[2], env[1], env[3])    # Interleaved
        # Shapely
        bbox = geom.bounds    # Interleaved: minX, minY, maxX, maxY

#        subdf = self.df.loc[self.rtree.intersection(bbox)]
#        
#        # For items passing the rough cut, see if the geometries actually intersect
#        if row[self.shapecol].Intersection(geom):
#            subrows.append(ix)

        subrows = list()
        for ix in self.idx.intersection(bbox):
            row = self.df.loc[ix]
            # For items passing teh rough cut, see if the geometries actually intersect
            if row[self.shapecol].intersection(geom):
                subrows.append(ix)

        return self.df.loc[subrows]

