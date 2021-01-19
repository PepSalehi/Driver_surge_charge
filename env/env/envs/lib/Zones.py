import numpy as np
import pandas as pd

# import geopandas as gpd
from collections import deque, defaultdict
# from functools import lru_cache
from lib.Requests import Req
from lib.Constants import WARMUP_TIME_SECONDS, BONUS, POLICY_UPDATE_INTERVAL, DEMAND_UPDATE_INTERVAL
from lib.Vehicles import VehState


def get_time_for_bookkeeping(t):
    time_used_for_bookkeeping_rewards = np.ceil(
        (t - POLICY_UPDATE_INTERVAL) / POLICY_UPDATE_INTERVAL)  # every 15 mins
    return time_used_for_bookkeeping_rewards


class Zone:
    """
    Attributes:
        rs1: a seeded random generator for requests
        id: zone_id
        DD : daily demand (every trip)
        M: demand matrix
        D: demand volume (trips/hour)
        V: number of vehicles
        K: capacity of vehicles
        vehs: the list of vehicles
        N: number of requests

        mid: row number of the demand file
        reqs: the list of requests. This is for debugging and reporting. demand served would NOT be removed from this list
        rejs: the list of rejected requests
        distance_rejs: the list of requests rejected because the distance from O to D
            was below the distance threshold (not included in rejs)
        queue: requests in the queue
        assign: assignment method
    """

    def __init__(self, ID, rs=None):
        """
        Initializes a zone object.
        @param ID: (int) zone id
        @param rs: random seeder
        """
        if rs is None:
            seed1 = 10
            self.rs1 = np.random.RandomState(seed1)
        else:
            self.rs1 = rs
        self.id = ID
        self.demand = deque([])  # demand maybe should be a time-based dictionary?
        self.served_demand = []
        self.idle_vehicles = list()
        # self.busy_vehicles = list()
        self.incoming_vehicles = list()
        self.undecided_vehicles = list()
        self.fare = None
        self.reqs = []
        self.N = 0
        self.M = None
        self.DD = None
        self.D = None
        self.pickup_binned = None
        self.mid = 0
        self.surge = 1
        self.bonus = 0
        self.num_surge = 0  # number of times there was a surge pricing in the zone
        self.DEMAND_ELASTICITY = -0.6084  # https://www.nber.org/papers/w22627.pdf
        self.adjusted_demand_rate = None
        self._n_matched = 0
        self.revenue_generated = 0
        self.reward_dict = defaultdict(float)
        self.served_req_dict = defaultdict(float)
        self.denied_req_dict = defaultdict(float)
        #
        self._demand_history = []
        self._served_demand_history = []
        self._supply_history = []
        self._incoming_supply_history = []
        # for debugging
        self._time_demand = []

    def read_daily_demand(self, demand_df):
        """
        Updates the daily OD demand of this zone.
        @param pickup_df: df pick ups
        @param demand_df: df describing OD demand for all zones.
        @return: None
        """
        # self.DD = demand_df.query("PULocationID == {zone_id}".format(zone_id=self.id))
        self.DD = demand_df[demand_df["PULocationID"] == self.id]  ## OD data
        # self.pickup_binned = pickup_df[pickup_df["PULocationID"] == self.id]

    def calculate_demand_function(self, demand, surge):
        """
        Calculates demand as a function of current demand, elasticity, and surge.

        This should be a decreasing function of price 
        use elasticities instead 
        -0.6084 for NYC
        @param demand:
        @param surge (float): surge multiplier.
        @requires surge >= 1

        @return (float): new demand according to surge
        """
        base_demand = demand
        change = self.DEMAND_ELASTICITY * (
                surge - 1
        )  # percent change in price * elascticity
        new_demand = int((1 + change) * base_demand)  # since change is negative
        new_demand = np.max([0, new_demand])
        assert new_demand <= base_demand
        # print("surge was ", surge)
        # print("change in demand as calculated by elasticity ", change)

        return new_demand

    def join_incoming_vehicles(self, veh):
        """
        Adds incoming vehicle to list of incoming vehicles.
        @param veh (Vehicle)
        @return: None
        """
        try:
            assert veh not in self.incoming_vehicles
        except AssertionError:
            print(veh.locations)
            print(veh.zone.id)
            print(veh.ozone)
            print(veh.rebalancing)
            print(veh.time_to_be_available)
            raise AssertionError

        self.incoming_vehicles.append(veh)

    def join_undecided_vehicles(self, veh):
        """
        Adds vehicle to list of undecided vehicles.
        @param veh (Vehicle)
        """
        try:
            assert veh not in self.undecided_vehicles
        except AssertionError:
            print(veh.locations)
            print(veh.zone.id)
            print(veh.ozone)
            print(veh.idle)
            print(veh.rebalancing)
            print(veh.time_to_be_available)
            raise AssertionError

        self.undecided_vehicles.append(veh)

    def remove_veh_from_waiting_list(self, veh):
        """
        Removes vehicle from idle vehicles.
        @param veh (Vehicle)
        """
        if veh in self.idle_vehicles:
            self.idle_vehicles.remove(veh)

    def identify_idle_vehicles(self):
        """
        Updates the idle vehicles and incoming vehicle list.
        """
        for v in self.incoming_vehicles:
            # if v.time_to_be_available <= 0:  # not v.rebalancing and
            if v._state == VehState.IDLE:
                assert v not in self.idle_vehicles

                self.idle_vehicles.append(v)
                self.incoming_vehicles.remove(v)

    def match_veh_demand(self, Zones, t, WARMUP_PHASE, operator, penalty=-10):
        """
        Matches idle vehicles to requests via a queue.
        @param Zones:
        @param t: time
        @param WARMUP_PHASE (bool)
        @param penalty (float)
        @return: None
        """

        # if there are no idle vehicles but there is demand, how do we take care of that?
        for v in self.idle_vehicles[:]:
            if len(self.demand) > 0:
                # check see if it's time
                if self.demand[0].Tr <= t:
                    req = self.demand.popleft()
                    req.set_surge_value(self.get_surge_multiplier())
                    status = v.match_w_req(req, Zones, WARMUP_PHASE)
                    self.calculate_reward(req, status, t, v)

        if len(self.demand) > 0:
            del_idx = []
            before_len = len(self.demand)
            demand_list = []
            excess_waits = []

            for req in self.demand:
                if req.has_exceeded_waiting_time(t):
                    self.denied_req_dict[get_time_for_bookkeeping(t)] += 1
                    # print("request {} in zone {} at time {} had excess wait with wait={} ".format(req.id, self.id, t,
                    #                                                                               req.get_waiting_time(
                    #                                                                                   t)))
                    excess_waits.append(req)
                else:
                    demand_list.append(req)

            self.demand = deque(demand_list)

    def calculate_reward(self, req, status, t, v):
        time_used_for_bookkeeping_rewards = get_time_for_bookkeeping(t)
        # np.ceil(t / POLICY_UPDATE_INTERVAL)  # every 15 mins
        if status:  # if matched, remove from the zone's idle list
            self._n_matched += 1
            before_len = len(self.idle_vehicles)
            self.idle_vehicles.remove(v)  # does this correctly remove the vehicle?
            assert len(self.idle_vehicles) < before_len
            assert v.ozone == req.dzone
            req.Tp = t
            # if not WARMUP_PHASE:
            self.served_demand.append(req)
            self.revenue_generated += req.get_effective_fare()
            # TODO: this is the total fare. Operator only gets a fraction of this
            self.reward_dict[time_used_for_bookkeeping_rewards] += req.get_effective_fare()
            self.served_req_dict[time_used_for_bookkeeping_rewards] += 1
            # operator.budget -= self.bonus

        else:  # when does this happen?
            print("Not matched by zone ", self.id)
            if v.is_AV:
                "should be penalized"
                # v.profits.append(penalty)

    # break

    def assign(self, Zones, t, WARMUP_PHASE, penalty, operator):
        """
        Identifies idle vehicles, then amends history and matches vehicle demand.

        @param Zones:
        @param t:
        @param WARMUP_PHASE:
        @param penalty:
        @return: None
        """
        self.identify_idle_vehicles()
        # bookkeeping
        self._demand_history.append(len(self.demand))
        self._served_demand_history.append(len(self.served_demand))
        self._supply_history.append(len(self.idle_vehicles))
        self._incoming_supply_history.append(len(self.incoming_vehicles))
        #
        self.match_veh_demand(Zones, t, WARMUP_PHASE, operator, penalty)

    def set_demand_rate_per_t(self, t):
        """
        Sets the demand per time period.
        This should use self.demand as the (hourly) demand, and then generate demand according to a Poisson distribution
        @param t: seconds
        """
        t_15_min = np.floor(t / DEMAND_UPDATE_INTERVAL)
        # demand = self.DD.query("Hour == {T}".format(T=t))
        self.this_t_demand = self.DD[self.DD['time_of_day_index_15m'] == t_15_min]
        self.D = self.this_t_demand.shape[0]  # number of rows, i.e., transactions

    def set_bonus(self, bonus):
        """
        Sets the driver bonus for serving a zone
        :param bonus: (float)
        :return: None
        """
        self.bonus = bonus

    def set_surge_multiplier(self, m):
        """
        Sets the surge multiplier.
        @param m: (float) desired surge multiplier
        """
        self.surge = m

    def get_surge_multiplier(self):
        return self.surge

    def generate_performance_stats(self, t):
        time_used_for_bookkeeping_rewards = get_time_for_bookkeeping(t)
        w = len(self.demand)  # current demand
        u = len(self.idle_vehicles)  # current supply
        served = self.served_req_dict[time_used_for_bookkeeping_rewards]
        denied = self.denied_req_dict[time_used_for_bookkeeping_rewards]
        total_demand = self.D
        un_served_demand = total_demand - served
        # assert unserved_demand >= 0
        # los = served / (served + w) if (served + w) > 0 else 0
        los = served / total_demand if total_demand > 0 else 0

        return w, u, total_demand, served, un_served_demand, denied, los

    # @profile
    def _generate_request(self, d, t_15):
        """
        Generate one request, following exponential arrival interval.
        https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter13_stochastic/02_poisson.ipynb
        @param d: demand (number)
        @return: request
        when would it return None??
            1. where there is no demand
            2. when it cannot find the demand info in the df
        """
        # check if there is any demand first
        if self.D == 0:  # i.e., no demand
            print("zone {} has no demand at t_15: {}".format(self.id, t_15))
            return

        time_interval = DEMAND_UPDATE_INTERVAL  # 15 minutes. TODO: make sure this imports from constants.py
        t_15_start = t_15 * time_interval
        # print("t_15_start :", t_15_start)
        rate = d
        scale = time_interval / d
        # inter-arrival time is generated according to the exponential distribution
        # y is the arrival times, between 0-time_interval
        y = np.cumsum(self.rs1.exponential(scale,
                                           size=int(rate)))
        y += t_15_start
        # print("arrival times are :", y)
        self.__generate_exact_req_object(y)

    def __generate_exact_req_object(self, y):
        # self.mid = 0
        # print("mid", self.mid)
        # def __generate_exact_req_object(self, y):

        two_d_array = self.this_t_demand.iloc[0:len(y)][["DOLocationID", "fare_amount"]].values
        two_d_array[:, 1] = two_d_array[:, 1] * self.surge + self.bonus
        id_counter_start = 0 if self.N == 0 else self.reqs[-1].id + 1
        ids = [id_counter_start + i for i in range(y.shape[0])]
        reqs = [Req(
            id=ids[i],
            Tr=y[i],
            ozone=self.id,
            dzone=int(two_d_array[i][0]),
            fare=two_d_array[i][1]
        )
            for i in range(y.shape[0])]

        self.reqs.extend(reqs)
        self.demand.extend(reqs)

    # @profile
    def generate_requests_to_time(self, t):
        """
        Generate requests up to time T, following Poisson process
        @param t: time (seconds)
        @return: None
        """
        t_15_min = np.floor(t / 900)

        self.set_demand_rate_per_t(t)
        # print("demand before possible surge in zone", self.id, self.D)
        demand = self.calculate_demand_function(self.D, self.surge)
        # print("demand after possible surge in zone", self.id, demand)

        before_demand = len(self.demand)
        # print("self.D", self.D)
        # print("demand after surge computation", demand)
        self._generate_request(self.D, t_15_min)

# TODO: df_hourly_stats_over_days is what a professional driver knows
# TODO: df_hourly_stats is stats per hour per day. Can be the true information provided by the operator (although how are they gonna know it in advance?)
