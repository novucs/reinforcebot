import React from 'react';
import {BASE_URL, displayErrors, ensureSignedIn, fetchMe, getAuthorization} from '../Util';
import {Button, Form, Grid, Header, Message, Segment} from "semantic-ui-react";
import logo from "../icon.svg";
import TopMenu from "../components/TopMenu";
import Footer from "../Footer";
import {SemanticToastContainer, toast} from "react-semantic-toasts";

export default class Profile extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      me: {},
      password: null,
      errors: [],
    }
  }

  submit = () => {
    let details = {
      id: this.state.me?.id,
      first_name: this.state.me?.first_name,
      last_name: this.state.me?.last_name,
      username: this.state.me?.username,
      email: this.state.me?.email,
    };
    if (this.state.password !== null) {
      details['password'] = this.state.password;
    }

    fetch(BASE_URL + '/api/auth/users/me/', {
      method: 'PATCH',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...getAuthorization(),
      },
      body: JSON.stringify(details),
    }).then(response => {
      if (response.status !== 200) {
        response.json().then(body => {
          this.setState({
            errors: displayErrors(
              body['first_name'],
              body['last_name'],
              body['username'],
              body['email'],
              body['password'])
          });
        });
        return;
      }

      this.setState({errors: []});
      response.json().then(body => {

        console.log(body);
      });

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Profile details successfully updated</p>
        },
      );
    });
  };

  keyPress = (event) => {
    if (event.keyCode === 13 && this.ableToSubmit()) {
      this.submit(event);
    }
  };

  ableToSubmit = () => {
    return this.state.firstName !== ''
      && this.state.lastName !== ''
      && this.state.username !== ''
      && this.state.email !== ''
      && this.state.password !== '';
  };

  componentDidMount = () => {
    ensureSignedIn();
    fetchMe(me => this.setState({me}));
  };

  render = () => (
    <div className='SitePage'>
      <TopMenu me={this.state.me}/>
      <SemanticToastContainer position='bottom-right'/>
      <Grid textAlign='center' style={{marginTop: '32px', marginBottom: '32px'}} className='SiteContents'
            verticalAlign='middle'>
        <Grid.Column style={{maxWidth: 450}}>
          <Header as="h2" color="teal" textAlign="center">
            <img src={logo} alt="logo" className="image"/>
            {this.state.me?.username} / Profile
          </Header>
          <Form size="large">
            <Segment stacked>
              <Form.Input
                fluid
                defaultValue={this.state.me?.first_name}
                placeholder="First name"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({me: {...this.state.me, first_name: event.target.value}})}
              />
              <Form.Input
                fluid
                defaultValue={this.state.me?.last_name}
                placeholder="Last name"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({me: {...this.state.me, last_name: event.target.value}})}
              />
              <Form.Input
                fluid
                defaultValue={this.state.me?.email}
                icon="envelope"
                iconPosition="left"
                placeholder="Email"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({me: {...this.state.me, email: event.target.value}})}
              />
              <Form.Input
                fluid
                defaultValue={this.state.me?.username}
                icon="user"
                iconPosition="left"
                placeholder="Username"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({me: {...this.state.me, username: event.target.value}})}
              />
              <Form.Input
                fluid
                icon="lock"
                iconPosition="left"
                placeholder="Password"
                type="password"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({password: event.target.value})}
              />
              <Button
                color="teal"
                fluid size="large"
                disabled={!this.ableToSubmit()}
                onClick={this.submit}
              >
                Edit Details
              </Button>
            </Segment>
          </Form>
          <Message
            error
            header='Profile Change Unsuccessful'
            list={this.state.errors}
            hidden={this.state.errors.length === 0}
          />
        </Grid.Column>
      </Grid>
      <Footer/>
    </div>
  );
}
