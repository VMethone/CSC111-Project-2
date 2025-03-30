from __future__ import annotations
from collections import defaultdict
import pandas as pd
from ast import literal_eval

class Location():
    """
    Stores the location of an airport
    """
    location_id: int
    city_name: str
    year: int
    quarter: int
    geo_loc: tuple[float, float]
    airport_codes: set[str]
    def __init__(self, loc_id: int, name: str, geo_loc) -> None:
        self.location_id = loc_id
        self.city_name = name
        self.geo_loc = geo_loc
        self.airport_codes = []

    def __str__(self) -> str:
        return f"({self.city_name}: {self.location_id})"
    
    def __repr__(self) -> str:
        return f"({self.city_name}: {self.location_id})"

class Route():
    """
    A connection between two locations
    """
    route_id: str
    depart_loc: Location
    arrival_loc: Location
    year: int
    quarter: int
    dist: float
    fare: float
    def __init__(self, route_id: str, departure_city: Location, arrival_city: Location, 
                 year: int, quarter: int, dist: float, fare: float):
        self.depart_loc = departure_city
        self.arrival_loc = arrival_city
        self.route_id = route_id
        self.year = year
        self.quarter = quarter
        self.dist = dist
        self.fare = fare

    def __str__(self):
        return f"Route: {self.depart_loc} -> {self.arrival_loc} for ${self.fare} in {self.year} (Q{self.quarter})"

    def __repr__(self):
        return f"Route: {self.depart_loc} -> {self.arrival_loc} for ${self.fare} in {self.year} (Q{self.quarter})"

class Flights():
    """
    Parent class storing flight information
    """
    id_to_city: dict[int, Location]
    cities: set[Location]
    flight_routes: dict[Location, list[Route]]
    def __init__(self, csv: str=""):
        self.cities = set()
        self.flight_routes = defaultdict(list)
        self.id_to_city = defaultdict(Location)
        if csv:
            self.load_from_cvs(csv)

    def load_from_cvs(self, csv_path: str):
        """
        Load csv file into the Flights class extracting relevant information
        """
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            # city_id, city_name = row_dict['citymarketid_1'], row_dict['city1']


            self.add_city(row_dict['citymarketid_1'], row_dict["city1"], 
                          literal_eval(row_dict["Geocoded_City1"]), row_dict["airport_1"])
            self.add_city(row_dict['citymarketid_2'], row_dict["city2"], 
                          literal_eval(row_dict["Geocoded_City2"]), row_dict["airport_2"])

            start_location: Location = self.id_to_city[row_dict['citymarketid_1']]
            end_location: Location = self.id_to_city[row_dict['citymarketid_2']]

            self.add_route(
                route_id=row_dict['tbl1apk'],
                depart_loc=start_location,
                arrival_loc=end_location,
                year=row_dict['year'],
                quarter=row_dict['quarter'],
                dist=row_dict['nsmiles'],
                fare=row_dict['fare']
            )

    def get_location_from_name(self, city_name: str) -> Location:
        for loc in self.cities:
            if city_name in loc.city_name:
                print(f"Found location: {loc}")
                return loc
        
        return None
    
    def get_location_from_airport_code(self, code: str) -> Location:
        for loc in self.cities:
            if code in loc.airport_codes:
                return loc

    def add_city(self, city_id: int, city_name: str, location: tuple[int, int], code: str) -> None:
        """
        Add a new location to the list of locations
        """
        if not city_id in self.id_to_city:
            new_city = Location(city_id, city_name, location, code)
            self.id_to_city[city_id] = new_city
            self.cities.add(new_city)
            return
        
        else:
            self.id_to_city[city_id].airport_codes.add(code)

    def add_route(self, route_id: str, depart_loc: Location, arrival_loc: Location,
                   year: int, quarter: int, dist: float, fare: float) -> None:
        """
        Add at route between two locations
        """
        self.flight_routes[depart_loc].append(Route(
            route_id=route_id, departure_city=depart_loc, arrival_city=arrival_loc,
            year=year, quarter=quarter, dist=dist, fare=fare
        ))
        self.flight_routes[arrival_loc].append(Route(
            route_id=route_id, departure_city=arrival_loc, arrival_city=depart_loc,
            year=year, quarter=quarter, dist=dist, fare=fare
        ))
