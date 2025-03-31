"""
This module defines data classes for modeling airport locations, routes, and flight connections.
It includes functionality to load and parse airline route data from CSV files into structured objects.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd
from ast import literal_eval


class Location:
    """
    Stores the location of an airport
    """
    location_id: int
    city_name: str
    year: int
    quarter: int
    geo_loc: tuple[float, float]
    airport_codes: set[str]

    def __init__(self, loc_id: int, name: str, geo_loc: tuple[float, float], code: str) -> None:
        self.location_id = loc_id
        self.city_name = name
        self.geo_loc = geo_loc
        self.airport_codes = {code}

    def __str__(self) -> str:
        return f"({self.city_name}: {self.location_id})"

    def __repr__(self) -> str:
        return f"({self.city_name}: {self.location_id})"


class Route:
    """
    A connection between two locations
    """
    route_id: str
    depart_loc: Location
    arrival_loc: Location
    depart_airport: str
    arrival_airport: str
    year: int
    quarter: int
    dist: float
    fare: float

    def __init__(self, route_id: str, departure_city: tuple[Location, str], arrival_city: tuple[Location, str],
                 year: int, quarter: int, dist: float, fare: float) -> None:
        self.depart_loc, self.depart_airport = departure_city
        self.arrival_loc, self.arrival_airport = arrival_city
        self.route_id = route_id
        self.year = year
        self.quarter = quarter
        self.dist = dist
        self.fare = fare

    @staticmethod
    def get_route_path_string(all_routes: list[Route]) -> str:
        """
        Return a string representation of the route like: "CityA → CityB → CityC"
        """
        output = []
        output.append(f"({all_routes[0].depart_loc.city_name}: {all_routes[0].depart_airport})")
        for route in all_routes:
            output.append(f"({route.arrival_loc.city_name}: {route.arrival_airport})")

        return " → ".join(output)

    def __str__(self) -> str:
        return f"Route: {self.depart_loc} -> {self.arrival_loc} for ${self.fare} in {self.year} (Q{self.quarter})"

    def __repr__(self) -> str:
        return f"Route: {self.depart_loc} -> {self.arrival_loc} for ${self.fare} in {self.year} (Q{self.quarter})"


class Flights:
    """
    Parent class storing flight information
    """
    id_to_city: dict[int, Location]
    cities: set[Location]
    flight_routes: dict[Location, list[Route]]

    def __init__(self, csv: str = "") -> None:
        self.cities = set()
        self.flight_routes = defaultdict(list)
        self.id_to_city = {}
        if csv:
            self.load_from_cvs(csv)

    def load_from_cvs(self, csv_path: str) -> None:
        """
        Load csv file into the Flights class extracting relevant information
        """
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()

        for _, row in df.iterrows():
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
                depart_loc=(start_location, row_dict['airport_1']),
                arrival_loc=(end_location, row_dict['airport_2']),
                year=row_dict['year'],
                quarter=row_dict['quarter'],
                dist=row_dict['nsmiles'],
                fare=row_dict['fare']
            )

    def get_location_from_name(self, city_name: str) -> Location:
        """
            Returns the Location object matching the given city name.

            Parameters:
                - city_name (str): The name (or partial name) of the city to look for.

            Returns:
                - Location: The corresponding Location object if found; otherwise, None.
        """
        for loc in self.cities:
            if city_name in loc.city_name:
                return loc
        return None

    def add_city(self, city_id: int, city_name: str, location: tuple[int, int], code: str) -> None:
        """
        Add a new location to the list of locations
        """
        if city_id not in self.id_to_city:
            new_city = Location(city_id, city_name, location, code)
            self.id_to_city[city_id] = new_city
            self.cities.add(new_city)
            return
        else:
            self.id_to_city[city_id].airport_codes.add(code)

        if city_name != self.id_to_city[city_id]:
            pass
            # print(f"Found Duplicate for {city_name}: {city_name} != {self.id_to_city[city_id]}")

    def add_route(self, route_id: str, depart_loc: tuple[Location, str],
                  arrival_loc: tuple[Location, str], year: int, quarter: int, dist: float, fare: float) -> None:
        """
        Add at route between two locations
        """
        self.flight_routes[depart_loc[0]].append(Route(
            route_id=route_id, departure_city=depart_loc, arrival_city=arrival_loc,
            year=year, quarter=quarter, dist=dist, fare=fare
        ))
        self.flight_routes[arrival_loc[0]].append(Route(
            route_id=route_id, departure_city=arrival_loc, arrival_city=depart_loc,
            year=year, quarter=quarter, dist=dist, fare=fare
        ))

    def __repr__(self) -> str:
        output = []
        for city in self.cities:
            for route in self.flight_routes[city]:
                output.append(str(route))
        return "".join(output)


if __name__ == '__main__':
    # When you are ready to check your work with python_ta, uncomment the following lines.
    # (In PyCharm, select the lines below and press Ctrl/Cmd + / to toggle comments.)
    # You can use "Run file in Python Console" to run both pytest and PythonTA,
    # and then also test your methods manually in the console.
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ["collections", 'pandas', 'ast'],  # the names (strs) of imported modules
        'allowed-io': [],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120,
        'disable': ["E0611", "R0902", "R0913", "C0411"]
    })
