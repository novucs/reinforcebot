import React from 'react';
import {Route, Switch} from 'react-router-dom';

import Home from '../pages/Home';
import Register from '../pages/Register';

const Main = () => {
    return (
        <Switch>
            <Route exact path='/' component={Home}/>
            <Route exact path='/register' component={Register}/>
        </Switch>
    );
};

export default Main;
