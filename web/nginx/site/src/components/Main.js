import React from 'react';
import {Route, Switch} from 'react-router-dom';

import Home from '../pages/Home';
import SignUp from '../pages/SignUp';
import SignIn from '../pages/SignIn';
import Dashboard from "../pages/Dashboard";
import Start from "../pages/Start";
import AgentDetail from "../pages/AgentDetail";

const Main = () => {
  return (
    <Switch>
      <Route exact path='/' component={Home}/>
      <Route exact path='/start' component={Start}/>
      <Route exact path='/signup' component={SignUp}/>
      <Route exact path='/signin' component={SignIn}/>
      <Route exact path='/dashboard' component={Dashboard}/>
      <Route path='/agent/:id' component={AgentDetail}/>
    </Switch>
  );
};

export default Main;
