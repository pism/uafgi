import osgeo.ogr

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
        self.rtree = rtree.index.Index(interleaved=True)
        for ix,row in df.iterrows():    # (index, Series) pair
            geom = row[shapecol]
            env = geom.GetEnvelope()    # (minX, maxX, minY, maxY)
            bbox = (env[0], env[2], env[1], env[3])    # Interleaved
            idx.insert(ix, bbox)

    def intersection(self, geom):
        """Returns idex values of items intersecting with geom
        geom:
            OGR Geometry
        """

        # Do a rough cut...
        subrows = list()    # Index of rows we want to keep
        env = geom.GetEnvelope()
        bbox = (env[0], env[2], env[1], env[3])    # Interleaved
        for ix in self.rtree.intersection(bbox):
            row = self.df[ix]
            # For items passing teh rough cut, see if the geometries actually intersect
            if row[self.shapecol].Intersection(geom):
                subrows.append(ix)

        return subrows


