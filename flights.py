from __future__ import annotations
from collections import defaultdict
import pandas as pd
class Location():
    """
    Stores the location of an airport
    """
    location_id: int
    city_name: str
    geo_loc: tuple[float, float]
    def __init__(self, loc_id: int, name: str) -> None:
        self.location_id = loc_id
        self.city_name = name
        pass

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

class Flights():
    """
    Parent class storing flight information
    """
    id_to_city: dict[int, Location]
    cities: set[Location]
    flight_routes: dict[Location, list[Route]]
    def __init__(self):
        self.cities = set()
        self.flight_routes = defaultdict(list)
        pass

    def load_from_cvs(self, csv_path: str):
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            # city_id, city_name = row_dict['citymarketid_1'], row_dict['city1']
            self._add_city(row_dict['citymarketid_1'], row_dict["city1"])
            self._add_city(row_dict['citymarketid_2'], row_dict["city2"])

            start_location = self.id_to_city(row_dict['citymarketid_1'])
            end_location = self.id_to_city(row_dict['citymarketid_2'])

            self._add_route(
                route_id=row_dict['tbl1apk'],
                depart_loc=start_location,
                arrival_loc=end_location,
                year=row_dict['year'],
                quarter=row_dict['quarter'],
                dist=row_dict['nsmiles'],
                fare=row_dict['fare']
            )


    def _add_city(self, city_id: int, city_name: str) -> None:
        """
        Add a new location to the list of locations
        """
        if not city_id in self.id_to_city:
            self.id_to_city[city_id] = city_name
            self.cities.add(Location(city_id, city_name))
            return
        
        
        if city_name != self.id_to_city[city_id]:
            print(f"Found Duplicate for {city_name}: {city_name} != {self.id_to_city[city_id]}")

    def _add_route(self, route_id: str, depart_loc: Location, arrival_loc: Location,
                   year: int, quarter: int, dist: float, fare: float) -> None:
        
        self.flight_routes[depart_loc].append(Route(
            route_id=route_id, departure_city=depart_loc, arrival_city=arrival_loc,
            year=year, quarter=quarter, dist=dist, fare=fare
        ))
