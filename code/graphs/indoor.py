import xml.etree.ElementTree as ET
from collections import defaultdict

import networkx as nx
from shapely import geometry

GML = 'http://www.opengis.net/gml/3.2'
INDOOR = 'http://www.opengis.net/indoorgml/1.0/core'
NAVI = 'http://www.opengis.net/indoorgml/1.0/navigation'
XLINK = 'http://www.w3.org/1999/xlink'
ALMA = 'http://www.idsia.ch/alma'
SVG = 'http://www.w3.org/2000/svg'
GMLNS = '{%s}' % GML
INDOORNS = '{%s}' % INDOOR
NAVINS = '{%s}' % INDOOR
XLINKNS = '{%s}' % XLINK
ALMANS = '{%s}' % ALMA
SVGNS = '{%s}' % SVG

my_namespaces = {INDOOR: 'indoorCore', GML: "gml", XLINK: 'xlink',
                 ALMA: 'alma', NAVI: 'indoorNavi'}
ns = {v: k for k, v in my_namespaces.items()}

ET._namespace_map.update(my_namespaces)


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


def coord_to_gml(shape):
    return " ".join(["%.1f %.1f" % p for p in shape.coords])


def point_to_gml(point, id):
    gml = ET.Element("gml:Point", {'srsDimension': '2', 'gml:id': id})
    pos = ET.SubElement(gml, "gml:pos")
    pos.text = coord_to_gml(point)
    return gml


def line_to_gml(line, id):
    gml = ET.Element("gml:LineString", {'gml:id': id})
    pos = ET.SubElement(gml, "gml:posList")
    pos.text = coord_to_gml(line)
    return gml


def ring_to_gml(ring):
    gml = ET.Element("gml:LinearRing")
    pos = ET.SubElement(gml, "gml:posList")
    pos.text = coord_to_gml(ring)
    return gml


def polygon_to_gml(polygon, id):
    gml = ET.Element("gml:Polygon", {'gml:id': id})
    exterior = ET.SubElement(gml, "gml:exterior")
    exterior.append(ring_to_gml(polygon.exterior))
    for ring in polygon.interiors:
        i = ET.SubElement(gml, "gml:interior")
        i.append(ring_to_gml(ring))
    return gml


def coordinatesFromGML(p):
    node = p.find('.//gml:pos', ns)
    pos = node.text
    point = [float(n) for n in pos.split()[:2]]
    return tuple(point)


_ns2types = {}
_ns2types['indoorCore'] = ['CellSpace', 'CellSpaceBoundary', 'SpaceLayer',
                           'MultiLayeredGraph']
_ns2types['indoorNavi'] = ['NavigableSpace', 'NavigableBoundary',
                           'GeneralSpace', 'TransferSpace', 'TransitionSpace',
                           'AnchorSpace', 'ConnectionSpace']


class GMLFeature(object):
    """A minimal GML feature"""
    def __init__(self, id):
        super(GMLFeature, self).__init__()
        self.id = id
        self.name = ''
        self.description = ''
        self.type = 'GMLFeature'

    def __repr__(self):
        return ("%s: %s %s %s" %
                (self.type, self.id,
                 '(' + self.name + ')' if self.name else '',
                 '- ' + self.description if self.description else ''))

    @staticmethod
    def readMetaData(node, name):
        return node.find(".//{name}".format(name=name), ns)

    @classmethod
    def loadFromXMLNode(cls, node):
        if node is None:
            return None
        i = node.get('%sid' % GMLNS)

        instance = cls(i)
        t = node.tag.split('}')
        instance.type = t.pop()

        nameNode = node.find('gml:name', ns)
        if(nameNode is not None):
            instance.name = nameNode.text

        descNode = node.find('gml:description', ns)
        if(descNode is not None):
            instance.description = descNode.text

        return instance


class State(GMLFeature):
    """State is modeled after an indoorGML State"""
    def __init__(self, id):
        super(State, self).__init__(id)
        self.connects = []
        self.geometry = None
        self.duality = None
        self.attribute = {}
        self.equals = defaultdict(set)
        self.within = defaultdict(set)
        self.contains = defaultdict(set)
        self.overlaps = defaultdict(set)
        # OR of equals within contains overlaps
        self.intersects = defaultdict(set)
        # NOT in indoorGML
        self.touches = defaultdict(set)
        self.type = 'State'
        self.layer = None
        self.default = {}
        self.up = set()
        self.down = set()

    @property
    def neighbors(self):
        if not self.layer.graph.has_node(self.id):
            return []
        return [self.layer.states.get(s) for s in self.layer.graph[self.id]]

    def transitionsTo(self, state):
        if not self.layer.graph.has_node(self.id):
            return None
        ts = self.layer.graph[self.id].get(state.id, {}).values()
        return [self.layer.transitions.get(t['id']) for t in ts]

    @classmethod
    def loadFromXMLNode(cls, node):
        if(node is None):
            return None
        state = super(cls, cls).loadFromXMLNode(node)
        geometryNode = node.find('indoorCore:geometry//gml:pos', ns)

        if(geometryNode is not None):
            pos = node.find('indoorCore:geometry//gml:pos', ns).text
            point = [float(n) for n in pos.split()[:2]]
            state.geometry = geometry.Point(tuple(point))

        state.connects = []

        cellXML = node.find('indoorCore:duality/*', ns)
        state.duality = Cell.loadFromXMLNode(cellXML)
        if(state.duality):
            state.duality.duality = state
        for n in node.findall("gml:name", ns):
            state.name = n.text

        for mt in ['open', 'nontraversable', 'invisible']:
            m = GMLFeature.readMetaData(node, 'alma:{mt}'.format(mt=mt))
            if m is not None:
                state.default[mt] = True
            else:
                state.default[mt] = False
        return state


class Transition(GMLFeature):
    """Transition is modeled after an indoorGML State"""
    def __init__(self, id):
        super(Transition, self).__init__(id)
        self.geometry = None
        self.duality = None
        self.type = 'Transition'

    @property
    def connects(self):
        return [self.start, self.end]

    @classmethod
    def loadFromXMLNode(cls, node):
        if(node is None):
            return None
        transition = super(cls, cls).loadFromXMLNode(node)
        connects = [s.get(XLINKNS + 'href')[1:]
                    for s in node.findall('indoorCore:connects', ns)]
        transition.start = connects[0]
        transition.end = connects[1]

        transition.duality = (node.find('indoorCore:duality', ns).
                              get(XLINKNS + 'href')[1:])

        line = []
        for pos in node.findall('indoorCore:geometry//gml:pos', ns):
            line.append(tuple([float(n) for n in pos.text.split()[:2]]))

        for posList in node.findall('.//gml:posList', ns):
            coord = [float(n) for n in posList.text.split()]
            line = zip(coord[::2], coord[1::2])
        line = list(line)
        if(len(line) > 1):
            transition.geometry = geometry.LineString(line)

        return transition


class Cell(GMLFeature):
    """Transition is modeled after an indoorGML CellSpace"""
    def __init__(self, id):
        super(Cell, self).__init__(id)
        self.boundary = []
        self.geometry = None
        self.type = 'CellSpace'
        self.outer_edge = None
        self.inner_edges = []
        # NavigableSpace attributes:
        self.usage = None
        self.className = None
        self.function = None

    def edges(self):
        return [self.outer_edge] + self.inner_edges

    def addBoundary(self, b):
        self.boundary.append(b)
        b.addCell(self)

    def removeBoundary(self, b):
        if b in self.boundary:
            self.boundary.remove(b)
            b.removeCell(self)

    @classmethod
    def loadFromExternalReferenceNode(cls, externalRef):
        raise NameError("Not implemented yet")

    @classmethod
    def loadFromXMLNode(cls, node):
        if(node is None):
            return None
        externalRef = node.find('indoorCore:externalReference', ns)
        if(externalRef is not None):
            return cls.loadFromExternalReferenceNode(externalRef)

        cell = super(cls, cls).loadFromXMLNode(node)

        if cell.type != 'CellSpace':
            for n in node.findall('indoorNavi:class', ns):
                cell.className = n.text
            for n in node.findall('indoorNavi:usage', ns):
                cell.usage = n.text
            for n in node.findall('indoorNavi:function', ns):
                cell.function = n.text

        cell.boundary = []

        for boundaryNode in node.findall('indoorCore:partialboundedBy', ns):
            ref = boundaryNode.get(XLINKNS + 'href')
            if(ref is None):
                try:
                    cell.addBoundary(Boundary.loadFromXMLNode(boundaryNode[0]))
                except Exception as e:
                    pass
            else:
                cell.boundary.append(ref[1:])

        polygonXML = node.find('indoorCore:Geometry2D/gml:Polygon', ns)
        if(polygonXML is None):
            cell.geometry = None
            cell.boundary = []
        else:
            interior = []
            exterior = []

            for pos in polygonXML.findall('gml:exterior//gml:pos', ns):
                exterior.append(tuple([float(n) for n
                                       in pos.text.split()][:2]))

            for posList in polygonXML.findall('gml:exterior//gml:posList', ns):
                coord = [float(n) for n in posList.text.split()]
                exterior = zip(coord[::2], coord[1::2])

            for loop in polygonXML.findall('gml:interior//gml:LinearRing', ns):
                ls = []
                for pos in loop.findall('.//gml:pos', ns):
                    ls.append(tuple([float(n) for n in pos.text.split()][:2]))

                for posList in loop.findall('.//gml:posList', ns):
                    coord = [float(n) for n in posList.text.split()]
                    ls = zip(coord[::2], coord[1::2])

                interior.append(geometry.LinearRing(ls))

            cell.geometry = geometry.Polygon(exterior, interior)
            if not cell.geometry.is_valid:
                raise Exception("Invalid Cell %s: %s" %
                                (cell.id, cell.geometry.wkt))

        return cell


class Boundary(GMLFeature):
    """Transition is modeled after an indoorGML CellSpaceBoundary"""
    def __init__(self, id):
        super(Boundary, self).__init__(id)
        self.geometry = None
        self.duality = None
        self.type = 'CellSpaceBoundary'
        self._chains = {}
        self.cells = []

    def addCell(self, cell):
        self.cells.append(cell)

    def removeCell(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)

    @classmethod
    def loadFromXMLNode(cls, node):
        if(node is None):
            return None

        boundary = super(cls, cls).loadFromXMLNode(node)
        line = []
        poss = node.findall('indoorCore:geometry2D/gml:LineString/gml:pos', ns)

        for pos in poss:
                line.append(tuple([float(n) for n in pos.text.split()][:2]))

        posLists = node.findall(
            'indoorCore:geometry2D/gml:LineString/gml:posList', ns)
        for posList in posLists:
            coord = [float(n) for n in posList.text.split()]
            line = zip(coord[::2], coord[1::2])

        boundary.geometry = geometry.LineString(line)
        if not boundary.geometry.is_valid:
            raise Exception("Invalid Boundary %s %s" %
                            (boundary.id, boundary.geometry.wkt))
        return boundary


class Layer(GMLFeature):
    """Layer is modeled after an indoorGML SpaceLayer"""
    def __init__(self, id):
        super(Layer, self).__init__(id)
        self.className = ''
        self.usage = ''
        self.function = ''
        self.states = {}
        self.transitions = {}
        self.cells = {}
        self.boundaries = {}
        self.indexId = 0
        self.graph = nx.MultiGraph()
        self.type = 'SpaceLayer'
        self.map = None

    def state_with_name(self, name):
        states = [s for s in self.states.values() if s.name == name]
        if len(states):
            return states[0]
        return None

    def addState(self, state):
        state.layer = self
        self.states[state.id] = state
        if(state.duality):
            cell = state.duality
            self.cells[cell.id] = cell
            cell.layer = self

            if cell.geometry:
                self.indexId += 1

            if(cell.boundary):
                for boundary in cell.boundary[:]:
                    if(isinstance(boundary, str)):  # xlink:href
                        cell.boundary.remove(boundary)
                        o_boundary = self.boundaries.get(boundary, None)
                        if o_boundary:
                            cell.addBoundary(o_boundary)
                        else:
                            pass
                    else:
                        self.boundaries[boundary.id] = boundary

    def connectTransition(self, transition):
        if(transition.start.geometry and transition.end.geometry):
            transition.geometry = geometry.LineString(
                [transition.start.geometry,
                 transition.duality.geometry.centroid,
                 transition.end.geometry])

        transition.start.connects.append(transition)
        transition.end.connects.append(transition)

        if(transition.geometry):
            transition.weight = transition.geometry.length
        else:
            transition.weight = 0

        self.graph.add_edge(transition.start.id, transition.end.id,
                            transition.id, id=transition.id,
                            weight=transition.weight)

    def addTransitionWithBoundary(self, start, end, border):
        ntransition = Transition('%sT%d' %
                                 (self.id, len(self.transitions) + 1))
        ntransition.duality = border
        border.duality = ntransition
        self.transitions[ntransition.id] = ntransition
        ntransition.start = start
        ntransition.end = end
        self.connectTransition(ntransition)
        return ntransition

    def addTransition(self, transition):
        transition.layer = self
        transition.start = self.states.get(transition.start, None)

        if not transition.start:
            return

        transition.start.connects.append(transition)
        t_end = transition.end
        transition.end = self.states.get(t_end, None)

        transition.duality = self.boundaries[transition.duality]
        transition.duality.duality = transition
        self.transitions[transition.id] = transition

        if(transition.geometry):
            transition.weight = transition.geometry.length
        else:
            transition.weight = 0

        if not transition.end:
            pass
        else:
            transition.end.connects.append(transition)
            self.graph.add_edge(transition.start.id, transition.end.id,
                                transition.id, id=transition.id,
                                weight=transition.weight)

    def find_bounds(self):
        self.geometry = geometry.MultiPolygon(
            [s.duality.geometry for s in self.states.values() if s.duality])
        self.bounds = self.geometry.bounds

    @classmethod
    def loadFromXMLNode(cls, node):
        layer = super(cls, cls).loadFromXMLNode(node)
        layer.graph = nx.MultiGraph()
        layer.states = {}
        layer.transitions = {}
        layer.cells = {}
        layer.boundaries = {}
        for n in node.findall('./indoorCore:class', ns):
            layer.className = n.text

        for n in node.findall('./indoorCore:usage', ns):
            layer.usage = n.text

        for n in node.findall('./indoorCore:function', ns):
            layer.function = n.text

        for n in node.findall("indoorCore:nodes//indoorCore:State", ns):
            layer.addState(State.loadFromXMLNode(n))

        for n in node.findall("indoorCore:edges//indoorCore:Transition", ns):
            layer.addTransition(Transition.loadFromXMLNode(n))

        layer.find_bounds()

        layer.external_states = [s for s in layer.states.values()
                                 if not s.geometry]

        return layer

    def addBoundary(self, type='CellSpaceBoundary'):
        b = Boundary('%sB%d' % (self.id, len(self.boundaries) + 1))
        b.type = type
        self.boundaries[b.id] = b
        return b

    def addBoundaryWithGeometry(self, geometry, type='CellSpaceBoundary'):
        b = self.addBoundary(type)
        b.geometry = geometry
        return b


class IndoorMap(GMLFeature):
    """IndoorMap is modeled after an indoorGML Multi Layered Graph"""
    def __init__(self, id):
        super(IndoorMap, self).__init__(id)
        self.space_layers = {}
        self.externalMaps = {}
        self.states = {}
        self.cells = {}
        self.transitions = {}
        self.boundaries = {}
        self.file = ''
        self.origin = (0, 0)
        self.angle = 0
        self.geometricLayer = None

    def find_bounds(self):
        layers = self.space_layers.values()
        layers = list(layers)
        if(len(layers)):
            self.geometry = layers[0].geometry
            for l in layers[1:]:
                pass
            self.bounds = self.geometry.bounds
        else:
            self.geometry = None
            self.bounds = None

    def addLayer(self, layer):
        self.space_layers[layer.id] = layer
        layer.map = self

    def addInterEdgeFromXMLNode(self, node):

        n = node.find("indoorCore:typeOfTopoExpression", ns)

        if n is None:
            return

        t = node.find("indoorCore:typeOfTopoExpression", ns).text
        if(t != "CONTAINS"):
            return

        startId = node.find("indoorCore:start", ns).get('%shref' % XLINKNS)[1:]
        endId = node.find("indoorCore:end", ns).get('%shref' % XLINKNS)[1:]

        startLayers = [l for l in self.space_layers.values()
                       if l.has_node(startId)]
        endLayers = [l for l in self.space_layers.values()
                     if l.has_node(endId)]

        if(len(startLayers) != 1):
            raise Exception("Inter layer Connection %s not well formed. "
                            "Start Node %s  in layers %s" %
                            (node, startLayers, startId))
        if(len(endLayers) != 1):
            raise Exception("Inter layer Connection %s not well formed. "
                            "End Node %s  in layers %s" %
                            (node, endLayers, endId))

        startState = startLayers[0].node[startId]
        endState = endLayers[0].node[endId]
        startState.setdefault("contains", []).append(endState)
        endState.setdefault("contained_in", []).append(startState)

    @classmethod
    def loadFromFile(cls, file_name):
        """Load a multi layered graph from an indoorGML document"""
        tree = ET.parse(file_name)
        node = tree.getroot()
        if not node.tag == INDOORNS + "MultiLayeredGraph":
            node = node.find("indoorCore:MultiLayeredGraph", ns)
        if node is not None:
            m = cls.loadFromXMLNode(node)
            m.file = file_name
            return m
        else:
            raise Exception('Malformed xml file: no MultiLayeredGraph tag')

    @classmethod
    def loadFromXMLNode(cls, node):

        mlg = super(cls, cls).loadFromXMLNode(node)

        mlg.space_layers = {}
        for n in node.findall(".//indoorCore:SpaceLayer", ns):
            mlg.addLayer(Layer.loadFromXMLNode(n))
        for n in node.findall(".//indoorCore:interEdges", ns):
            mlg.addInterEdgeFromXMLNode(n)

        mlg.states = {}
        mlg.cells = {}
        mlg.boundaries = {}
        mlg.transitions = {}
        for l in mlg.space_layers.values():
            mlg.states.update(l.states)
            mlg.cells.update(l.cells)
            mlg.boundaries.update(l.boundaries)
            mlg.transitions.update(l.transitions)

        mlg.find_bounds()
        return mlg
